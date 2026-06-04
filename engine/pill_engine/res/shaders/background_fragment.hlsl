struct SkyCam {
    float3 right;     float tan_half_fov;
    float3 up;        float aspect;
    float3 fwd;       float _pad;
    float3 bg_color;  float _pad2;
};
[[vk::binding(0, 0)]] ConstantBuffer<SkyCam> UCam;
[[vk::binding(1, 0)]] Texture2D    texEquirect;
[[vk::binding(2, 0)]] SamplerState smpEquirect;

#include "include/equirect.hlsl"

[shader("fragment")]
float4 fs_main(float2 ndc : TEXCOORD0) : SV_Target {
    float3 dir = normalize(
        UCam.right * (ndc.x * UCam.tan_half_fov * UCam.aspect)
      + UCam.up    * (ndc.y * UCam.tan_half_fov)
      + UCam.fwd
    );
    float2 uv = dir_to_equirect_uv(dir);
    float3 sky = texEquirect.Sample(smpEquirect, uv).rgb;
    return float4(sky * UCam.bg_color, 1.0);
}
