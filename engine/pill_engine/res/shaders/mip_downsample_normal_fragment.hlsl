// Normal-map mip: averaged tangent-space normals shorten, so renormalize before re-encoding.
[[vk::binding(0, 0)]] Texture2D    source_texture;
[[vk::binding(1, 0)]] SamplerState source_sampler;

[shader("fragment")]
float4 fs_main(float2 texture_coordinates : TEXCOORD0) : SV_Target {
    float4 texel  = source_texture.Sample(source_sampler, texture_coordinates);
    float3 normal = texel.xyz * 2.0 - 1.0;
    float  length_squared = dot(normal, normal);
    normal = length_squared > 1e-8 ? normal * rsqrt(length_squared) : float3(0.0, 0.0, 1.0);
    return float4(normal * 0.5 + 0.5, texel.w);
}
