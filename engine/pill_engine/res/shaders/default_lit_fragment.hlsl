// Default lit fragment shader. Edit here — `pill_assets` regenerates the .wgsl.
// Diffuse + normal mapped, two lights, exp-squared depth fog.

#include "include/common.hlsl"

struct MaterialParams {
    float3 tint;
    float  _pad; // Rust renderer writes Color as 16 bytes (xyz + 1 float pad); spec lands at offset 16
    float  spec;
};
[[vk::binding(0, 2)]] ConstantBuffer<MaterialParams> material;

[[vk::binding(0, 3)]] Texture2D    diffuse_texture;
[[vk::binding(1, 3)]] SamplerState diffuse_sampler;
[[vk::binding(2, 3)]] Texture2D    normal_texture;
[[vk::binding(3, 3)]] SamplerState normal_sampler;

[shader("fragment")]
float4 fs_main(
    [[vk::location(0)]] float3 in_vertex_position       : TEXCOORD0,
    [[vk::location(1)]] float2 in_vertex_texture_coords : TEXCOORD1,
    [[vk::location(2)]] float3 in_TBN_tangent           : TEXCOORD2,
    [[vk::location(3)]] float3 in_TBN_bitangent         : TEXCOORD3,
    [[vk::location(4)]] float3 in_TBN_normal            : TEXCOORD4,
    [[vk::location(5)]] float3 in_world_position        : TEXCOORD5
) : SV_TARGET {
    // Key light: blue, diffuse-dominant, high above-front-left.
    const float  ambient_strength  = 0.28;
    const float3 key_dir           = float3(-4.0, 12.0, -10.0);
    const float3 key_color         = float3(0.45, 0.55, 1.0);

    // Spec light: white, tight highlight, from lower-right-front for a cross-axis streak.
    const float3 spec_dir          = float3(6.0, -3.0, -8.0);
    const float3 spec_color        = float3(1.0, 1.0, 1.0);
    const float  spec_exponent     = 154.0;

    float4 object_color  = diffuse_texture.Sample(diffuse_sampler, in_vertex_texture_coords);
    float4 object_normal = normal_texture.Sample(normal_sampler, in_vertex_texture_coords);

    // TBN is tangent-to-world (vertex shader stores the transpose).
    float3x3 TBN_matrix     = float3x3(in_TBN_tangent, in_TBN_bitangent, in_TBN_normal);
    float3   normal_tangent = normalize(object_normal.rgb * 2.0 - 1.0);
    float3   normal         = normalize(mul(TBN_matrix, normal_tangent));

    float3 view_direction = normalize(camera.camera_position - in_world_position);

    // Key light — diffuse + soft specular.
    float3 key_light_dir     = normalize(key_dir);
    float  key_diffuse       = max(dot(normal, key_light_dir), 0.0);
    float3 key_half          = normalize(view_direction + key_light_dir);
    float  key_spec          = pow(max(dot(normal, key_half), 0.0), 24.0) * material.spec * 0.4;
    float3 key_contribution  = key_color * (ambient_strength + key_diffuse * 1.2);

    // Spec light — specular only, crisp white streak.
    float3 spec_light_dir    = normalize(spec_dir);
    float3 spec_half         = normalize(view_direction + spec_light_dir);
    float  spec_strength     = pow(max(dot(normal, spec_half), 0.0), spec_exponent) * material.spec * 3.0;
    float3 spec_contribution = spec_color * spec_strength;

    float3 final_color = key_contribution * object_color.xyz * material.tint
                       + (key_color * key_spec + spec_contribution);

    // Exponential-squared depth fog. density = 0 → no fog, bit-identical output.
    float fog_dist   = length(camera.camera_position - in_world_position);
    float fog_factor = clamp(1.0 - exp(-engine.fog_density * engine.fog_density * fog_dist * fog_dist), 0.0, 1.0);
    final_color = lerp(final_color, engine.fog_color, fog_factor);

    return float4(final_color, 1.0);
}
