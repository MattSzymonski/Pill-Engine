// Per-pixel gather along the pixel's own motion vector (McGuire et al. 2012 without the
// tile-max neighbourhood). Samples across a depth discontinuity are down-weighted.

struct MotionBlurParams {
    float velocity_scale; // shutter fraction of the frame interval, 0 = off
    float min_speed;      // pixels; slower pixels are copied
    float depth_softness; // weight falloff per unit of view-space depth difference
    uint  max_samples;
    float depth_mul;      // view distance = depth_mul / (depth_add - depth)
    float depth_add;
    float2 _padding;
};

[[vk::binding(0, 0)]] Texture2D<float4> color_texture;
[[vk::binding(1, 0)]] Texture2D<float4> velocity_texture;
[[vk::binding(2, 0)]] Texture2D<float>  depth_texture;
[[vk::binding(3, 0)]] ConstantBuffer<MotionBlurParams> params;
[[vk::binding(4, 0)]] SamplerState linear_sampler;

static const int MAX_SAMPLES = 64;

float view_distance(float2 texture_coordinates, uint2 size) {
    int2  texel = clamp(int2(texture_coordinates * float2(size)), int2(0, 0), int2(size) - 1);
    float depth = depth_texture.Load(int3(texel, 0));
    return params.depth_mul / (params.depth_add - depth);
}

[shader("fragment")]
float4 fs_main(float2 texture_coordinates : TEXCOORD0) : SV_Target {
    uint width, height;
    velocity_texture.GetDimensions(width, height);
    uint2 size = uint2(width, height);

    float2 velocity     = velocity_texture.SampleLevel(linear_sampler, texture_coordinates, 0).xy * params.velocity_scale;
    float  speed_pixels = length(velocity * float2(size));
    float3 center       = color_texture.SampleLevel(linear_sampler, texture_coordinates, 0).rgb;
    if (speed_pixels < params.min_speed) {
        return float4(center, 1.0);
    }

    float center_distance = view_distance(texture_coordinates, size);
    int   sample_count    = clamp(int(speed_pixels), 2, min(int(params.max_samples), MAX_SAMPLES));

    float3 accumulated = center;
    float  weight_sum  = 1.0;
    for (int index = 0; index < sample_count; index++) {
        // Shutter centered on the frame: offsets span -0.5..0.5 of the motion vector
        float  offset          = float(index) / float(sample_count - 1) - 0.5;
        float2 sample_location = texture_coordinates + velocity * offset;
        float  sample_distance = view_distance(sample_location, size);
        float  weight          = 1.0 / (1.0 + abs(sample_distance - center_distance) * params.depth_softness);
        accumulated += color_texture.SampleLevel(linear_sampler, sample_location, 0).rgb * weight;
        weight_sum  += weight;
    }
    return float4(accumulated / weight_sum, 1.0);
}
