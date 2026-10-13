// One mip level from the previous one: a bilinear tap at the 2×2 block centre averages it.
[[vk::binding(0, 0)]] Texture2D    source_texture;
[[vk::binding(1, 0)]] SamplerState source_sampler;

[shader("fragment")]
float4 fs_main(float2 texture_coordinates : TEXCOORD0) : SV_Target {
    return source_texture.Sample(source_sampler, texture_coordinates);
}
