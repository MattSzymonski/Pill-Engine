struct VOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

VOut vs_main(uint vi : SV_VertexID) {
    float2 p[3] = {
        float2(-1.0, -3.0),
        float2( 3.0,  1.0),
        float2(-1.0,  1.0)
    };
    VOut o;
    o.pos = float4(p[vi], 0.0, 1.0);
    o.uv  = float2(p[vi].x * 0.5 + 0.5, -p[vi].y * 0.5 + 0.5);
    return o;
}
