// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/buffer_cache/buffer.h"

namespace VideoCore {

struct ImageInfo;
struct Image;
class StreamBuffer;

class TileManager {
    static constexpr size_t NUM_BPPS = 5;

public:
    using ScratchBuffer = std::pair<vk::Buffer, VmaAllocation>;
    using Result = std::pair<vk::Buffer, u32>;

    explicit TileManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         StreamBuffer& stream_buffer);
    ~TileManager();

    /// Converts a linear buffer region into the guest tiled layout described by info.
    /// Both buffers must support storage-buffer access. Input and output may be
    /// non-overlapping regions of the same Vulkan buffer.
    void TileBuffer(const ImageInfo& info, vk::Buffer linear_buffer, u32 linear_offset,
                    vk::Buffer tiled_buffer, u32 tiled_offset);

    struct CpuTilingResult {
        u64 retiled{};
        u64 padded_tail{};
        u64 out_of_range{};
        u32 first_bad_x{};
        u32 first_bad_y{};
        u64 first_bad_offset{};
    };

    /// Returns true when the current CPU address implementation can reproduce
    /// the guest tiled layout represented by info.
    [[nodiscard]] static bool CanTileBufferCpu(const ImageInfo& info);

    /// Converts a linear host buffer into PS4 guest tiled layout using CPU-side
    /// GCN/GpuAddress equations. The caller supplies the full padded tiled span.
    [[nodiscard]] static CpuTilingResult TileBufferCpu(const ImageInfo& info,
                                                       const u8* linear, u8* tiled,
                                                       u32 tiled_span_size);

    void TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                   vk::Buffer out_buffer, u32 out_offset, u32 copy_size);

    Result DetileImage(vk::Buffer in_buffer, u32 in_offset, const ImageInfo& info);

private:
    vk::Pipeline GetTilingPipeline(const ImageInfo& info, bool is_tiler);
    ScratchBuffer GetScratchBuffer(u32 size);

private:
    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    StreamBuffer& stream_buffer;
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pl_layout;
    std::array<vk::UniquePipeline, AmdGpu::NUM_TILE_MODES * NUM_BPPS> detilers{};
    std::array<vk::UniquePipeline, AmdGpu::NUM_TILE_MODES * NUM_BPPS> tilers{};
};

} // namespace VideoCore
