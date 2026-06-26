#include <metal_stdlib>
using namespace metal;

struct UInt4Uniforms { uint a,b,c,d; };
struct Int8Uniforms { int a,b,c,d,e,f,g,h; };

static inline uint pcg_hash(uint v) {
    v = v * 747796405u + 2891336453u;
    uint word = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (word >> 22u) ^ word;
}
static inline float u01(uint v) { return (float(v >> 8) + 0.5f) * (1.0f / 16777216.0f); }
static inline int floor_div_int(int a, int b) { int q = a / b; int r = a % b; return (r != 0 && ((r > 0) != (b > 0))) ? q - 1 : q; }

kernel void td_fill_float(device float* dst [[buffer(0)]], constant float& value [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if (id < count) dst[id] = value; }

kernel void td_gaussian_nchw(device float* out [[buffer(0)]], constant UInt4Uniforms& dims [[buffer(1)]], constant Int8Uniforms& origin [[buffer(2)]], constant uint& seedLo [[buffer(3)]], constant uint& seedHi [[buffer(4)]], constant float& sigma [[buffer(5)]], uint3 gid [[thread_position_in_grid]]) {
    int x=int(gid.x), y=int(gid.y), z=int(gid.z); int C=int(dims.a), H=int(dims.b), W=int(dims.c), N=int(dims.d);
    if (x>=W || y>=H || z>=C*N) return; int n=z/C, c=z-n*C;
    int gy = origin.a + y; int gx = origin.b + x; int th=max(origin.c,1), tw=max(origin.d,1);
    int ty=floor_div_int(gy,th), tx=floor_div_int(gx,tw); int ly=gy-ty*th, lx=gx-tx*tw;
    uint h = seedLo ^ (seedHi * 0x9E3779B9u) ^ uint(ty)*0x85ebca6bu ^ uint(tx)*0xc2b2ae35u ^ uint(c)*0x27d4eb2du ^ uint(n)*0x165667b1u;
    uint idx = uint((ly*tw + lx) * max(C,1) + c);
    float u1 = max(u01(pcg_hash(h ^ (idx*2u+0u))), 1e-12f); float u2 = u01(pcg_hash(h ^ (idx*2u+1u)));
    float r = sqrt(-2.0f * log(u1)); float a = 6.28318530718f * u2;
    out[((n*C+c)*H+y)*W+x] = sigma * r * cos(a);
}

kernel void td_pack_linear_weight(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant UInt4Uniforms& dims [[buffer(2)]], constant float& eps [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    int x=int(gid.x), y=int(gid.y), z=int(gid.z); int C=int(dims.a), H=int(dims.b), W=int(dims.c), N=int(dims.d); int PC=C+1;
    if (x>=W || y>=H || z>=PC*N) return; int n=z/PC, c=z-n*PC;
    float midY=float(H-1)*0.5f, midX=float(W-1)*0.5f;
    float wy = 1.0f - (1.0f - eps) * clamp(abs(float(y)-midY)/max(midY,1e-6f), 0.0f, 1.0f);
    float wx = 1.0f - (1.0f - eps) * clamp(abs(float(x)-midX)/max(midX,1e-6f), 0.0f, 1.0f);
    float w = wy * wx;
    output[((n*PC+c)*H+y)*W+x] = (c==C) ? w : input[((n*C+c)*H+y)*W+x] * w;
}

kernel void td_normalize_packed(device const float* packed [[buffer(0)]], device float* output [[buffer(1)]], constant UInt4Uniforms& dims [[buffer(2)]], constant float& eps [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    int x=int(gid.x), y=int(gid.y), z=int(gid.z); int C=int(dims.a), H=int(dims.b), W=int(dims.c), N=int(dims.d); if (x>=W||y>=H||z>=C*N) return; int n=z/C,c=z-n*C; int PC=C+1; float den=max(packed[((n*PC+C)*H+y)*W+x],eps); output[((n*C+c)*H+y)*W+x]=packed[((n*PC+c)*H+y)*W+x]/den;
}

kernel void td_accumulate_window(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant Int8Uniforms& d [[buffer(2)]], constant Int8Uniforms& e [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    int x=int(gid.x), y=int(gid.y), c=int(gid.z); int C=d.a, sH=d.b, sW=d.c, dH=d.d, dW=d.e, sy0=d.f, sx0=d.g, dy0=d.h, dx0=e.a, h=e.b, w=e.c; if (x>=w||y>=h||c>=C) return; dst[(c*dH+dy0+y)*dW+dx0+x] += src[(c*sH+sy0+y)*sW+sx0+x];
}

kernel void td_copy_region(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant Int8Uniforms& d [[buffer(2)]], constant Int8Uniforms& e [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    int x=int(gid.x), y=int(gid.y), c=int(gid.z); int C=d.a, sH=d.b, sW=d.c, dH=d.d, dW=d.e, sy0=d.f, sx0=d.g, dy0=d.h, dx0=e.a, h=e.b, w=e.c; if (x>=w||y>=h||c>=C) return; dst[(c*dH+dy0+y)*dW+dx0+x] = src[(c*sH+sy0+y)*sW+sx0+x];
}

kernel void td_copy_tile_to_batch(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant Int8Uniforms& d [[buffer(2)]], constant uint& b [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) { int x=int(gid.x), y=int(gid.y), c=int(gid.z); int C=d.a,H=d.b,W=d.c,N=d.d; if(x>=W||y>=H||c>=C||b>=uint(N)) return; dst[((int(b)*C+c)*H+y)*W+x]=src[(c*H+y)*W+x]; }

kernel void td_extract_batch_tile(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant Int8Uniforms& d [[buffer(2)]], constant uint& b [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) { int x=int(gid.x), y=int(gid.y), c=int(gid.z); int C=d.a,H=d.b,W=d.c,N=d.d; if(x>=W||y>=H||c>=C||b>=uint(N)) return; dst[(c*H+y)*W+x]=src[((int(b)*C+c)*H+y)*W+x]; }

kernel void td_unary_scale(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant float& scale [[buffer(2)]], constant float& bias [[buffer(3)]], constant uint& count [[buffer(4)]], uint id [[thread_position_in_grid]]) { if(id<count) dst[id]=src[id]*scale+bias; }

kernel void td_linear_mix(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* dst [[buffer(2)]], constant float& wa [[buffer(3)]], constant float& wb [[buffer(4)]], constant uint& count [[buffer(5)]], uint id [[thread_position_in_grid]]) { if(id<count) dst[id]=wa*a[id]+wb*b[id]; }

kernel void td_cat_channels(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* dst [[buffer(2)]], constant Int8Uniforms& d [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) { int x=int(gid.x), y=int(gid.y), z=int(gid.z); int ca=d.a, cb=d.b, H=d.c, W=d.d, N=d.e, C=ca+cb; if(x>=W||y>=H||z>=C*N) return; int n=z/C, c=z-n*C; if(c<ca) dst[((n*C+c)*H+y)*W+x]=a[((n*ca+c)*H+y)*W+x]; else {int bc=c-ca; dst[((n*C+c)*H+y)*W+x]=b[((n*cb+bc)*H+y)*W+x];} }

kernel void td_mp_silu(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if(id<count){ float x=src[id]; dst[id]=(x/(1.0f+exp(-x)))/0.596f; } }

kernel void td_signed_sqrt(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if(id<count){ float v=src[id]; dst[id]=sign(v)*sqrt(abs(v)); } }

kernel void td_inverse_signed_sqrt(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) { if(id<count){ float v=src[id]; dst[id]=sign(v)*v*v; } }

kernel void td_conv2d_nchw(device const float* src [[buffer(0)]], device const float* weight [[buffer(1)]], device float* dst [[buffer(2)]], constant Int8Uniforms& d0 [[buffer(3)]], constant Int8Uniforms& d1 [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
    int x=int(gid.x), y=int(gid.y), z=int(gid.z); int N=d0.a,Cin=d0.b,H=d0.c,W=d0.d,Cout=d0.e,KH=d0.f,KW=d0.g,groups=max(d0.h,1); int padY=d1.a,padX=d1.b,outH=d1.c,outW=d1.d,strideY=max(d1.e,1),strideX=max(d1.f,1); if(x>=outW||y>=outH||z>=Cout*N) return; int n=z/Cout, oc=z-n*Cout, group=oc*groups/Cout, cinPerGroup=Cin/groups, c0=group*cinPerGroup; float acc=0; for(int icg=0; icg<cinPerGroup; ++icg){ int ic=c0+icg; for(int ky=0; ky<KH; ++ky){ int sy=y*strideY+ky-padY; if(sy<0||sy>=H) continue; for(int kx=0; kx<KW; ++kx){ int sx=x*strideX+kx-padX; if(sx<0||sx>=W) continue; acc += src[((n*Cin+ic)*H+sy)*W+sx] * weight[((oc*cinPerGroup+icg)*KH+ky)*KW+kx]; } } } dst[((n*Cout+oc)*outH+y)*outW+x]=acc;
}

kernel void td_box_downsample(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant Int8Uniforms& d [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) { int x=int(gid.x), y=int(gid.y), c=int(gid.z); int C=d.a,H=d.b,W=d.c,outH=d.d,outW=d.e,fy=max(d.f,1),fx=max(d.g,1); if(x>=outW||y>=outH||c>=C) return; float s=0; int cnt=0; for(int yy=0; yy<fy; ++yy){int sy=y*fy+yy; if(sy>=H) continue; for(int xx=0; xx<fx; ++xx){int sx=x*fx+xx; if(sx>=W) continue; s+=src[(c*H+sy)*W+sx]; cnt++;}} dst[(c*outH+y)*outW+x]=s/max(float(cnt),1.0f); }

kernel void td_nearest_upsample(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant Int8Uniforms& d [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) { int x=int(gid.x),y=int(gid.y),c=int(gid.z); int C=d.a,H=d.b,W=d.c,outH=d.d,outW=d.e,fy=max(d.f,1),fx=max(d.g,1); if(x>=outW||y>=outH||c>=C) return; int sy=clamp(y/fy,0,H-1), sx=clamp(x/fx,0,W-1); dst[(c*outH+y)*outW+x]=src[(c*H+sy)*W+sx]; }

kernel void td_select_channel(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant UInt4Uniforms& d [[buffer(2)]], constant uint& channel [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) { int x=int(gid.x), y=int(gid.y); int C=int(d.a),H=int(d.b),W=int(d.c); if(x>=W||y>=H||channel>=uint(C)) return; dst[y*W+x]=src[(int(channel)*H+y)*W+x]; }

kernel void td_write_channel(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant UInt4Uniforms& d [[buffer(2)]], constant uint& channel [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) { int x=int(gid.x), y=int(gid.y); int C=int(d.a),H=int(d.b),W=int(d.c); if(x>=W||y>=H||channel>=uint(C)) return; dst[(int(channel)*H+y)*W+x]=src[y*W+x]; }
