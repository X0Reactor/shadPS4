// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/tile_manager.h"

#include "video_core/host_shaders/tiling_comp.h"

#include <magic_enum/magic_enum.hpp>
#include <vk_mem_alloc.h>

namespace VideoCore {

struct TilingInfo {
    u32 bank_swizzle;
    u32 num_slices;
    u32 num_mips;
    std::array<ImageInfo::MipInfo, 16> mips;
};

TileManager::TileManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         StreamBuffer& stream_buffer_)
    : instance{instance}, scheduler{scheduler}, stream_buffer{stream_buffer_} {
    const auto device = instance.GetDevice();
    const std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 2,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }};

    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto desc_layout_result = device.createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(desc_layout_result.result == vk::Result::eSuccess,
               "Failed to create descriptor set layout: {}",
               vk::to_string(desc_layout_result.result));
    desc_layout = std::move(desc_layout_result.value);

    const vk::DescriptorSetLayout set_layout = *desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 0U,
        .pPushConstantRanges = nullptr,
    };
    auto [layout_result, layout] = device.createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess, "Failed to create pipeline layout: {}",
               vk::to_string(layout_result));
    pl_layout = std::move(layout);
}

TileManager::~TileManager() = default;

TileManager::ScratchBuffer TileManager::GetScratchBuffer(u32 size) {
    constexpr auto usage =
        vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;

    const vk::BufferCreateInfo buffer_ci = {
        .size = size,
        .usage = usage,
    };

    const VmaAllocationCreateInfo alloc_info{
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };

    VkBuffer buffer;
    VmaAllocation allocation;
    const auto buffer_ci_unsafe = static_cast<VkBufferCreateInfo>(buffer_ci);
    const auto result = vmaCreateBuffer(instance.GetAllocator(), &buffer_ci_unsafe, &alloc_info,
                                        &buffer, &allocation, nullptr);
    ASSERT(result == VK_SUCCESS);
    return {buffer, allocation};
}

vk::Pipeline TileManager::GetTilingPipeline(const ImageInfo& info, bool is_tiler) {
    const u32 pl_id = u32(info.tile_mode) * NUM_BPPS + std::bit_width(info.num_bits) - 4;
    auto& tiling_pipelines = is_tiler ? tilers : detilers;
    if (auto pipeline = *tiling_pipelines[pl_id]; pipeline != VK_NULL_HANDLE) {
        return pipeline;
    }

    const auto device = instance.GetDevice();
    const auto micro_tile_mode = AmdGpu::GetMicroTileMode(info.tile_mode);
    std::vector<std::string> defines = {
        fmt::format("BITS_PER_PIXEL={}", info.num_bits),
        fmt::format("NUM_SAMPLES={}", info.num_samples),
        fmt::format("ARRAY_MODE={}", u32(info.array_mode)),
        fmt::format("MICRO_TILE_MODE={}", u32(micro_tile_mode)),
        fmt::format("MICRO_TILE_THICKNESS={}", AmdGpu::GetMicroTileThickness(info.array_mode)),
    };
    if (AmdGpu::IsMacroTiled(info.array_mode)) {
        const auto macro_tile_mode =
            AmdGpu::CalculateMacrotileMode(info.tile_mode, info.num_bits, info.num_samples);
        const u32 num_banks = AmdGpu::GetNumBanks(macro_tile_mode);
        defines.emplace_back(
            fmt::format("PIPE_CONFIG={}", u32(AmdGpu::GetPipeConfig(info.tile_mode))));
        defines.emplace_back(fmt::format("BANK_WIDTH={}", AmdGpu::GetBankWidth(macro_tile_mode)));
        defines.emplace_back(fmt::format("BANK_HEIGHT={}", AmdGpu::GetBankHeight(macro_tile_mode)));
        defines.emplace_back(fmt::format("NUM_BANKS={}", num_banks));
        defines.emplace_back(fmt::format("NUM_BANK_BITS={}", std::bit_width(num_banks) - 1));
        defines.emplace_back(fmt::format(
            "TILE_SPLIT_BYTES={}", AmdGpu::CalculateTileSplit(info.tile_mode, info.array_mode,
                                                              micro_tile_mode, info.num_bits)));
        defines.emplace_back(
            fmt::format("MACRO_TILE_ASPECT={}", AmdGpu::GetMacrotileAspect(macro_tile_mode)));
    }
    if (is_tiler) {
        defines.emplace_back(fmt::format("IS_TILER=1"));
    }

    const auto& module = Vulkan::Compile(HostShaders::TILING_COMP,
                                         vk::ShaderStageFlagBits::eCompute, device, defines);
    const auto module_name = fmt::format("{}_{} {}", magic_enum::enum_name(info.tile_mode),
                                         info.num_bits, is_tiler ? "tiler" : "detiler");
    LOG_INFO(Render_Vulkan, "Compiling shader {}", module_name);
    for (const auto& def : defines) {
        LOG_INFO(Render_Vulkan, "#define {}", def);
    }
    Vulkan::SetObjectName(device, module, module_name);
    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };
    const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
        .stage = shader_ci,
        .layout = *pl_layout,
    };
    auto [result, pipeline] =
        device.createComputePipelineUnique(VK_NULL_HANDLE, compute_pipeline_ci);
    ASSERT_MSG(result == vk::Result::eSuccess, "Detiler pipeline creation failed {}",
               vk::to_string(result));
    tiling_pipelines[pl_id] = std::move(pipeline);
    device.destroyShaderModule(module);
    return *tiling_pipelines[pl_id];
}

TileManager::Result TileManager::DetileImage(vk::Buffer in_buffer, u32 in_offset,
                                             const ImageInfo& info) {
    if (!info.props.is_tiled) {
        return {in_buffer, in_offset};
    }

    TilingInfo params{};
    params.bank_swizzle = info.bank_swizzle;
    params.num_slices = info.props.is_volume ? info.size.depth : info.resources.layers;
    params.num_mips = info.resources.levels;
    for (u32 mip = 0; mip < params.num_mips; ++mip) {
        auto& mip_info = params.mips[mip];
        mip_info = info.mips_layout[mip];
        if (info.props.is_block) {
            mip_info.pitch = std::max((mip_info.pitch + 3) / 4, 1U);
            mip_info.height = std::max((mip_info.height + 3) / 4, 1U);
        }
    }

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };

    const auto [out_buffer, out_allocation] = GetScratchBuffer(info.guest_size);
    scheduler.DeferOperation([this, out_buffer, out_allocation]() {
        vmaDestroyBuffer(instance.GetAllocator(), out_buffer, out_allocation);
    });

    scheduler.EndRendering();

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, GetTilingPipeline(info, false));

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = in_buffer,
        .offset = in_offset,
        .range = info.guest_size,
    };

    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = out_buffer,
        .offset = 0,
        .range = info.guest_size,
    };

    const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &tiled_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &linear_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &params_buffer_info,
        },
    }};
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pl_layout, 0, set_writes);

    const auto dim_x = (info.guest_size / (info.num_bits / 8)) / 64;
    cmdbuf.dispatch(dim_x, 1, 1);
    return {out_buffer, 0};
}

bool TileManager::CanTileBufferCpu(const ImageInfo& info) {
    if (!info.props.is_tiled || info.props.is_block || info.props.is_volume ||
        info.num_bits != 8 || info.num_samples != 1 ||
        info.resources.levels != 1 || info.resources.layers != 1 ||
        info.size.depth != 1 ||
        static_cast<u32>(info.tile_mode) != 0 ||
        static_cast<u32>(info.array_mode) != 4 ||
        static_cast<u32>(info.bank_swizzle) != 0) {
        return false;
    }

    const auto& mip = info.mips_layout[0];
    return mip.pitch != 0 && (mip.pitch % 256) == 0 &&
           mip.height != 0 && (mip.height % 128) == 0 &&
           info.guest_size != 0;
}

TileManager::CpuTilingResult TileManager::TileBufferCpu(const ImageInfo& info,
                                                        const u8* linear, u8* tiled,
                                                        u32 tiled_span_size) {
    CpuTilingResult result{};

    if (!CanTileBufferCpu(info) || linear == nullptr || tiled == nullptr ||
        tiled_span_size == 0) {
        return result;
    }

    // PS4/GCN 2D-thin depth/stencil layout currently validated here:
    //   bits=8, samples=1, arrayMode=4, microTileMode=depth,
    //   pipeConfig=P8_32x32_16x16, bankWidth=1, bankHeight=4,
    //   numBanks=16, tileSplitBytes=64, macroTileAspect=4.
    constexpr u32 kMicroTileWidth = 8;
    constexpr u32 kMicroTileHeight = 8;
    constexpr u32 kNumPipes = 8;
    constexpr u32 kNumBanks = 16;
    constexpr u32 kBankWidth = 1;
    constexpr u32 kBankHeight = 4;
    constexpr u32 kMacroTileAspect = 4;
    constexpr u32 kPipeInterleaveBytes = 256;
    constexpr u32 kPipeInterleaveBits = 8;
    constexpr u32 kPipeBits = 3;
    constexpr u32 kBankBits = 4;
    constexpr u32 kTileBytes = 64;
    constexpr u32 kMacroTileWidth =
        (kMicroTileWidth * kBankWidth * kNumPipes) * kMacroTileAspect;
    constexpr u32 kMacroTileHeight =
        (kMicroTileHeight * kBankHeight * kNumBanks) / kMacroTileAspect;
    constexpr u32 kMacroTileBytes =
        (kMacroTileWidth / kMicroTileWidth) *
        (kMacroTileHeight / kMicroTileHeight) *
        kTileBytes / (kNumPipes * kNumBanks);

    const auto element_index = [](u32 x, u32 y) -> u32 {
        // GpuAddress depth/thin microtile element order:
        // x0,y0,x1,y1,x2,y2.
        return (((x >> 0) & 1U) << 0) |
               (((y >> 0) & 1U) << 1) |
               (((x >> 1) & 1U) << 2) |
               (((y >> 1) & 1U) << 3) |
               (((x >> 2) & 1U) << 4) |
               (((y >> 2) & 1U) << 5);
    };

    const auto pipe_index = [](u32 x, u32 y) -> u32 {
        // Pipe configuration P8_32x32_16x16.
        return ((((x >> 3) ^ (y >> 3) ^ (x >> 4)) & 1U) << 0) |
               ((((x >> 4) ^ (y >> 4)) & 1U) << 1) |
               ((((x >> 5) ^ (y >> 5)) & 1U) << 2);
    };

    const auto bank_index = [](u32 x, u32 y) -> u32 {
        const u32 xs = x >> 3;
        const u32 ys = y >> 2;

        return ((((xs >> 3) ^ (ys >> 6)) & 1U) << 0) |
               ((((xs >> 4) ^ (ys >> 5) ^ (ys >> 6)) & 1U) << 1) |
               ((((xs >> 5) ^ (ys >> 4)) & 1U) << 2) |
               ((((xs >> 6) ^ (ys >> 3)) & 1U) << 3);
    };

    const auto& mip = info.mips_layout[0];

    const auto tiled_byte_offset = [&](u32 x, u32 y) -> u64 {
        const u64 elem = element_index(x, y);
        const u64 pipe = pipe_index(x, y);

        u64 bank = bank_index(x, y);
        bank ^= static_cast<u64>(info.bank_swizzle);
        bank &= (kNumBanks - 1);

        const u64 macro_tiles_per_row =
            static_cast<u64>(mip.pitch) / kMacroTileWidth;
        const u64 macro_tile_row_index = y / kMacroTileHeight;
        const u64 macro_tile_column_index = x / kMacroTileWidth;
        const u64 macro_tile_index =
            macro_tile_row_index * macro_tiles_per_row +
            macro_tile_column_index;
        const u64 macro_tile_offset =
            macro_tile_index * kMacroTileBytes;

        const u64 tile_row_index =
            (y / kMicroTileHeight) % kBankHeight;
        const u64 tile_column_index =
            ((x / kMicroTileWidth) / kNumPipes) % kBankWidth;
        const u64 tile_index =
            tile_row_index * kBankWidth + tile_column_index;
        const u64 tile_offset = tile_index * kTileBytes;

        const u64 total_byte_offset =
            macro_tile_offset + tile_offset + elem;

        const u64 pipe_interleave_offset =
            total_byte_offset & (kPipeInterleaveBytes - 1);
        const u64 offset =
            total_byte_offset >> kPipeInterleaveBits;

        return pipe_interleave_offset |
               (pipe << kPipeInterleaveBits) |
               (bank << (kPipeInterleaveBits + kPipeBits)) |
               (offset << (kPipeInterleaveBits +
                          kPipeBits + kBankBits));
    };

    bool have_bad = false;

    for (u32 y = 0; y < info.size.height; ++y) {
        for (u32 x = 0; x < mip.pitch; ++x) {
            const u64 linear_offset =
                static_cast<u64>(y) * mip.pitch + x;
            const u64 tiled_offset = tiled_byte_offset(x, y);

            if (tiled_offset >= tiled_span_size) {
                ++result.out_of_range;
                if (!have_bad) {
                    have_bad = true;
                    result.first_bad_x = x;
                    result.first_bad_y = y;
                    result.first_bad_offset = tiled_offset;
                }
                continue;
            }

            if (tiled_offset >= info.guest_size) {
                ++result.padded_tail;
            }

            tiled[tiled_offset] = linear[linear_offset];
            ++result.retiled;
        }
    }

    return result;
}

void TileManager::TileBuffer(const ImageInfo& info, vk::Buffer linear_buffer, u32 linear_offset,
                             vk::Buffer tiled_buffer, u32 tiled_offset) {
    ASSERT(info.props.is_tiled);
    ASSERT(info.num_bits >= 8);
    ASSERT(info.resources.levels > 0);

    TilingInfo params{};
    params.bank_swizzle = info.bank_swizzle;
    params.num_slices = info.props.is_volume ? info.size.depth : info.resources.layers;
    params.num_mips = info.resources.levels;

    u32 tiled_span_size = 0;
    for (u32 mip = 0; mip < params.num_mips; ++mip) {
        auto& mip_info = params.mips[mip];
        mip_info = info.mips_layout[mip];

        // Preserve the full padded tiled span from UpdateSize()/ImageInfo.
        // info.guest_size may intentionally be smaller and represents the real
        // guest-visible byte count that should be dispatched/written back.
        tiled_span_size =
            std::max(tiled_span_size, mip_info.offset + mip_info.size);

        if (info.props.is_block) {
            mip_info.pitch = std::max((mip_info.pitch + 3) / 4, 1U);
            mip_info.height = std::max((mip_info.height + 3) / 4, 1U);
        }
    }
    tiled_span_size = std::max(tiled_span_size, info.guest_size);

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };

    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();

    // Vulkan copy -> linear staging must be visible to the tiling compute shader.
    // The tiled output uses its full padded footprint even when only a smaller
    // guest-visible prefix is copied back afterward.
    const std::array<vk::BufferMemoryBarrier2, 2> pre_barriers = {{
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .buffer = linear_buffer,
            .offset = linear_offset,
            .size = info.guest_size,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask =
                vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .buffer = tiled_buffer,
            .offset = tiled_offset,
            .size = tiled_span_size,
        },
    }};

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });

    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, GetTilingPipeline(info, true));

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = tiled_buffer,
        .offset = tiled_offset,
        .range = tiled_span_size,
    };
    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = linear_buffer,
        .offset = linear_offset,
        .range = info.guest_size,
    };

    const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &tiled_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &linear_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &params_buffer_info,
        },
    }};

    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pl_layout, 0, set_writes);

    const u32 bytes_per_element = info.num_bits / 8;
    const u32 num_elements = info.guest_size / bytes_per_element;
    ASSERT(num_elements % 64 == 0);
    cmdbuf.dispatch(num_elements / 64, 1, 1);

    const vk::BufferMemoryBarrier2 post_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .buffer = tiled_buffer,
        .offset = tiled_offset,
        .size = tiled_span_size,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}
// Generic helper above is used by TextureCache for separate 8-bit stencil
// readback. No title-specific address, coordinate, stencil value or decision
// is encoded here.

void TileManager::TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                            vk::Buffer out_buffer, u32 out_offset, u32 copy_size) {
    const auto& info = in_image.info;
    if (!info.props.is_tiled) {
        for (auto& copy : buffer_copies) {
            copy.bufferOffset += out_offset;
        }
        in_image.Download(buffer_copies, out_buffer, out_offset, copy_size);
        return;
    }

    TilingInfo params{};
    params.bank_swizzle = info.bank_swizzle;
    params.num_slices = info.props.is_volume ? info.size.depth : info.resources.layers;
    params.num_mips = static_cast<u32>(buffer_copies.size());
    for (u32 mip = 0; mip < params.num_mips; ++mip) {
        auto& mip_info = params.mips[mip];
        mip_info = info.mips_layout[mip];
        if (info.props.is_block) {
            mip_info.pitch = std::max((mip_info.pitch + 3) / 4, 1U);
            mip_info.height = std::max((mip_info.height + 3) / 4, 1U);
        }
    }

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };

    const auto [temp_buffer, temp_allocation] = GetScratchBuffer(info.guest_size);
    scheduler.DeferOperation([this, temp_buffer, temp_allocation]() {
        vmaDestroyBuffer(instance.GetAllocator(), temp_buffer, temp_allocation);
    });

    const auto cmdbuf = scheduler.CommandBuffer();
    in_image.Download(buffer_copies, temp_buffer, 0, copy_size);

    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, GetTilingPipeline(info, true));

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = out_buffer,
        .offset = out_offset,
        .range = info.guest_size,
    };

    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = temp_buffer,
        .offset = 0,
        .range = info.guest_size,
    };

    const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &tiled_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &linear_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &params_buffer_info,
        },
    }};
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pl_layout, 0, set_writes);

    const auto dim_x = (info.guest_size / (info.num_bits / 8)) / 64;
    cmdbuf.dispatch(dim_x, 1, 1);
}

} // namespace VideoCore
