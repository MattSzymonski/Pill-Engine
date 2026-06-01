[[vk::binding(0, 0)]] Texture2D    texHdr;
[[vk::binding(1, 0)]] SamplerState smpHdr;

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD0) : SV_Target {
    float3 hdr    = texHdr.Sample(smpHdr, uv).rgb;
    float3 mapped = hdr / (hdr + 1.0);
    return float4(mapped, 1.0);
}
