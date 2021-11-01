#version 430

#ifdef COMPUTE_SHADER

layout(local_size_x = 16, local_size_y = 16) in;

//~ in vec2 vertex_texcoord;
//~ out vec4 fragment_color;
vec4 fragment_color;

uniform vec3 mouse;
//uniform vec3 motion;
uniform vec2 viewport;

uniform float iTime;

//uniform mat4 mvpMatrix;
uniform mat4 mvpInvMatrix;

layout(binding= 0, rgba32f) coherent uniform image2D image;

vec4 iMouse;
vec2 iResolution;
////////



#define DEBUG_PETAL false
#define CONTROL_CAMERA true
#define SKY true
#define SUN true
//
#define SEPARATE_CAUSTICS false
//
#define NEE false

vec2 cartesianToPolar(in vec2 p)
{
	float r = length(p);
	float theta = atan(p.y, p.x);
	return vec2(r, theta);
}
float max3(vec3 v)
{
	return max(max(v.x, v.y), v.z);
}
float min3(vec3 v)
{
	return min(min(v.x, v.y), v.z);
}
float sqr(float x)
{
	return x*x;
}

//-----------------------
const bool CORRELATED_SAMPLES = false;
//http://reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
//32 bits
uint rand_lcg(inout uint state)
{
	// LCG values from Numerical Recipes
	state = 1664525u * state + 1013904223u;
	return state;
}
//http://reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
//32 bits
uint rand_xorshift(inout uint state)
{
	// Xorshift algorithm from George Marsaglia's paper
	state ^= (state << 13);
	state ^= (state >> 17);
	state ^= (state << 5);
	return state;
}
//http://reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
void wang_hash(inout uint seed)
{
	seed = (seed ^ 61u) ^ (seed >> 16u);
	seed *= 9u;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2du;
	seed = seed ^ (seed >> 15);
}
uint randState = 0u;
void initRandState(const float time, const uvec2 pixel, const uvec2 resolution, const uint iteration)
{
	uint i = iteration;
	if(CORRELATED_SAMPLES)
		randState = i;
	else
		randState = pixel.x + resolution.x*(pixel.y + resolution.y*i);
	wang_hash(randState);
}
float randUniform(inout uint state)
{
	return fract(float(
		//rand_lcg(state)
		rand_xorshift(state)
	) * (1.0 / 4294967296.0));
}
float randUniform()
{
	return randUniform(randState);
}
//*/

//-----------------------



//https://www.shadertoy.com/view/4dffRH
vec3 hash( vec3 p ) // replace this by something better. really. do
{
	p = vec3( dot(p,vec3(127.1,311.7, 74.7)),
			  dot(p,vec3(269.5,183.3,246.1)),
			  dot(p,vec3(113.5,271.9,124.6)));
	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}
// return value noise (in x) and its derivatives (in yzw)
vec4 noised( in vec3 x )
{
	// grid
	vec3 i = floor(x);
	vec3 w = fract(x);
	
	#if 1
	// quintic interpolant
	vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
	vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
	#else
	// cubic interpolant
	vec3 u = w*w*(3.0-2.0*w);
	vec3 du = 6.0*w*(1.0-w);
	#endif
	
	// gradients
	vec3 ga = hash( i+vec3(0.0,0.0,0.0) );
	vec3 gb = hash( i+vec3(1.0,0.0,0.0) );
	vec3 gc = hash( i+vec3(0.0,1.0,0.0) );
	vec3 gd = hash( i+vec3(1.0,1.0,0.0) );
	vec3 ge = hash( i+vec3(0.0,0.0,1.0) );
	vec3 gf = hash( i+vec3(1.0,0.0,1.0) );
	vec3 gg = hash( i+vec3(0.0,1.0,1.0) );
	vec3 gh = hash( i+vec3(1.0,1.0,1.0) );
	
	// projections
	float va = dot( ga, w-vec3(0.0,0.0,0.0) );
	float vb = dot( gb, w-vec3(1.0,0.0,0.0) );
	float vc = dot( gc, w-vec3(0.0,1.0,0.0) );
	float vd = dot( gd, w-vec3(1.0,1.0,0.0) );
	float ve = dot( ge, w-vec3(0.0,0.0,1.0) );
	float vf = dot( gf, w-vec3(1.0,0.0,1.0) );
	float vg = dot( gg, w-vec3(0.0,1.0,1.0) );
	float vh = dot( gh, w-vec3(1.0,1.0,1.0) );
	
	// interpolations
	return vec4( va + u.x*(vb-va) + u.y*(vc-va) + u.z*(ve-va) + u.x*u.y*(va-vb-vc+vd) + u.y*u.z*(va-vc-ve+vg) + u.z*u.x*(va-vb-ve+vf) + (-va+vb+vc-vd+ve-vf-vg+vh)*u.x*u.y*u.z,	// value
				 ga + u.x*(gb-ga) + u.y*(gc-ga) + u.z*(ge-ga) + u.x*u.y*(ga-gb-gc+gd) + u.y*u.z*(ga-gc-ge+gg) + u.z*u.x*(ga-gb-ge+gf) + (-ga+gb+gc-gd+ge-gf-gg+gh)*u.x*u.y*u.z +   // derivatives
				 du * (vec3(vb,vc,ve) - va + u.yzx*vec3(va-vb-vc+vd,va-vc-ve+vg,va-vb-ve+vf) + u.zxy*vec3(va-vb-ve+vf,va-vb-vc+vd,va-vc-ve+vg) + u.yzx*u.zxy*(-va+vb+vc-vd+ve-vf-vg+vh) ));
}
//https://www.shadertoy.com/view/Xsl3Dl
float noise( in vec3 p )
{
	vec3 i = floor( p );
	vec3 f = fract( p );
	
	vec3 u = f*f*(3.0-2.0*f);

	return mix( mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
						  dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
					 mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
						  dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
				mix( mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
						  dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
					 mix( dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
						  dot( hash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

//https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
float sdRoundBox( vec3 p, vec3 b, float r )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}
float sdCylinder(vec3 p, vec3 a, vec3 b, float r)
{
	vec3  ba = b - a;
	vec3  pa = p - a;
	float baba = dot(ba,ba);
	float paba = dot(pa,ba);
	float x = length(pa*baba-ba*paba) - r*baba;
	float y = abs(paba-baba*0.5)-baba*0.5;
	float x2 = x*x;
	float y2 = y*y*baba;

	float d = (max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));

	return sign(d)*sqrt(abs(d))/baba;
}
float sdRoundedCylinder( vec3 p, float ra, float rb, float h )
{
  vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}
float dot2( in vec3 v ) { return dot(v,v); }
float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2)
{
	// sampling independent computations (only depend on shape)
	vec3  ba = b - a;
	float l2 = dot(ba,ba);
	float rr = r1 - r2;
	float a2 = l2 - rr*rr;
	float il2 = 1.0/l2;

	// sampling dependant computations
	vec3 pa = p - a;
	float y = dot(pa,ba);
	float z = y - l2;
	float x2 = dot2( pa*l2 - ba*y );
	float y2 = y*y*l2;
	float z2 = z*z*l2;

	// single square root!
	float k = sign(rr)*rr*rr*x2;
	if( sign(z)*a2*z2 > k ) return  sqrt(x2 + z2)		*il2 - r2;
	if( sign(y)*a2*y2 < k ) return  sqrt(x2 + y2)		*il2 - r1;
							return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}
float udQuad( vec3 p, vec3 a, vec3 b, vec3 c, vec3 d )
{
  vec3 ba = b - a; vec3 pa = p - a;
  vec3 cb = c - b; vec3 pb = p - b;
  vec3 dc = d - c; vec3 pc = p - c;
  vec3 ad = a - d; vec3 pd = p - d;
  vec3 nor = cross( ba, ad );

  return sqrt(
	(sign(dot(cross(ba,nor),pa)) +
	 sign(dot(cross(cb,nor),pb)) +
	 sign(dot(cross(dc,nor),pc)) +
	 sign(dot(cross(ad,nor),pd))<3.0)
	 ?
	 min( min( min(
	 dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
	 dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
	 dot2(dc*clamp(dot(dc,pc)/dot2(dc),0.0,1.0)-pc) ),
	 dot2(ad*clamp(dot(ad,pd)/dot2(ad),0.0,1.0)-pd) )
	 :
	 dot(nor,pa)*dot(nor,pa)/dot2(nor) );
}
float opSmoothUnion( float d1, float d2, float k )
{
	float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
	return mix( d2, d1, h ) - k*h*(1.0-h);
}
float opSmoothSubtraction( float d1, float d2, float k )
{
	float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
	return mix( d2, -d1, h ) + k*h*(1.0-h);
}
float opSmoothIntersection( float d1, float d2, float k )
{
	float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
	return mix( d2, d1, h ) + k*h*(1.0-h);
}




const float PI = 3.14;

#define SUN_DIRECTION normalize(vec3(-1.))
const float SUN_INTENSITY = 314.;
#define SUN_COLOR vec3(249, 231, 42)/256.


vec2 rotate(in float rads, in vec2 p)
{
	float c = cos(rads);
	float s = sin(rads);
	return vec2(
		c*p.x - s*p.y,
		s*p.x + c*p.y
	);
}
vec3 rotateZ(in float rads, in vec3 p)
{
	return vec3(rotate(rads, p.xy), p.z);
}
vec3 rotateX(in float rads, in vec3 p)
{
	return rotateZ(rads, p.yzx).zxy;
}
vec3 rotateY(in float rads, in vec3 p)
{
	return rotateZ(rads, p.zxy).yzx;
}

float sphereD(in vec3 p)
{
	return length(p) - 1.;
}
float sphereD(in vec3 c, in float r, in vec3 p)
{
	return length(p - c) - r;
}
//a disk of unit radius in the xy plane
float diskD(in vec3 p)
{
	float r2 = dot(p.xy, p.xy);
	//Projection in the disk, closest is disk surface.
	if(r2 < 1.0) return abs(p.z);
	
	//Projection out of disk, closest is distance to circle.
	float r = sqrt(r2);
	float h = r - 1.0;
	return sqrt(h*h + p.z*p.z);
}
vec3 diskP(in vec3 p)
{
	float r2 = dot(p.xy, p.xy);
	//Projection in the disk, closest is disk surface.
	if(r2 < 1.0) return vec3(p.xy, sign(p.z));
	
	//Projection out of disk, closest is distance to circle.
	float r = sqrt(r2);
	return vec3(cos(r), sin(r), sign(p.z));
}

void spiralify(inout vec2 p, in float a, in float b, in float falloffRadius)
{
	const float cst = 0.;
	float r = dot(p, p);
	if(r > falloffRadius*falloffRadius) return;
	r = sqrt(r);
	//float angle = atan(p.y/r, p.x/r);
	float angle = acos(p.x/r) * sign(p.y);
	float spiralangle = angle - sign(a) / b * log(r / abs(a) + cst);
	//float spiralangle = angle - sign(a) / b * log(r / abs(a) * min(1., r/falloffRadius));
	//float spiralangle = angle - sign(a) / b * log(r / abs(a) + cst) * (1. - r/falloffRadius);
	//p = mix(r * vec2(cos(spiralangle), sin(spiralangle)), p, r/falloffRadius);
	p = r * vec2(cos(spiralangle), sin(spiralangle));
}
void spiralifyElongation(inout vec2 p, in float radius, in float elongationFactor)
{
	vec2 mouseFactors = iMouse.xy/iResolution.xy;
	mouseFactors.y = 1. - mouseFactors.y;

	float a = -0.25 * (mouseFactors.x -0.5);
	float arcLength = abs(elongationFactor)*radius;
	float b = sign(elongationFactor)*radius/sqrt(arcLength*arcLength - radius*radius);

	spiralify(p, a, b, radius);
}
float spiralifyAngle(inout vec2 p, in float radius, in float angle)
{
	vec2 mouseFactors = iMouse.xy/iResolution.xy;
	mouseFactors.y = 1. - mouseFactors.y;

	float a = -0.25 * (mouseFactors.x -0.5);
	float b = log(radius/a) / angle;

	//p = rotate(-angle, p);
	//p.y -= -0.5*radius;
	spiralify(p, a, b, radius);
	//p.y += -0.5*radius;
	//p = rotate(angle, p);

	return b;
}

float spiralD(in vec2 p, in float a, in float b, in float radius)
{
	//calculate the target radius and theta
	float r = length(p);
	float t = atan(p.y, p.x);
	
	//early exit if the point requested is the origin itself
	//to avoid taking the logarithm of zero in the next step
	if(r == 0.)
		return 0.;
	
	//calculate the floating point approximation for n
	float n = (log(r/a)/b - t)/(2.*PI);
	
	//compute the endpoint
	float theta = log(radius/a)/b;
	vec2 endPoint = radius*vec2(cos(theta), sin(theta));
	float endPointD = distance(p, endPoint);
	
	float upper_r = a * exp(b * (t + 2.*PI*ceil(n)));
	if(upper_r > radius)
	{
		n = (log(radius/a)/b - t)/(2.*PI);
		float lower_r = a * exp(b * (t + 2.*PI*floor(n)));
		return min(endPointD, abs(r - lower_r));
	}
	
	//find the two possible radii for the closest point
	float lower_r = a * exp(b * (t + 2.*PI*floor(n)));
	
	//return the minimum distance to the target point
	return min(endPointD, min(abs(upper_r - r), abs(r - lower_r)));
}


//----------------------------------------------------
#define TABLE 0u
#define FLASK 1u
#define LABEL 2u
#define PISTIL 3u
#define PETAL 4u
#define SEPAL 5u
#define STEM 6u
#define LEAF 7u
#define CAP 8u

struct Hit
{
	float d;
	vec3 uv;
	vec3 p;
	uint m;
	vec3 n;
};

//maturity in [0, 2]
Hit petal(in float r, in float maturity, in vec3 p, in vec3 wp, in uint i)
{
	maturity = 1.;
	//maturity = 2.*(1.-iMouse.y/iResolution.y);

	////Small spiral at the end of the petal.
	//p.xz -= vec2(0.3, -0.);
	//spiralify(p.xz, 13.3);
	////TODO: find why approximatively 13.3 ?
	//p.xz += vec2(0.3, -0.);
	//Big spiral at the origin of the petal.
	//float scaleFactor = 3.*(1. - iMouse.y/iResolution.y);
	//p.xz /= scaleFactor;
	////spiralify(p.xz, 13.3); //approx 13.3 with small spiral
	//spiralify(p.xz, 6.);
	//p.xz *= scaleFactor;
	//
	//spiralifyAngle(p.xz, 0.1, -2.*PI);
	//scale
	vec3 scaling = r*vec3(1, 0.5, 1.);
	p /= scaling;
	//float b = spiralifyAngle(p.xz, 2., -2.*PI);
	//spiralifyElongation(p.xz, 3., 1.1);
	//rotate
	p = rotateY((-0.5 + 0.5*maturity)*PI, p);
	//p = rotateY(-atan(1./b), p);
	//shift
	p.x -= 1.;
	//p.x -= 0.1;
	//wang_hash(i);
	//p.z += 0.1*randUniform(i); //displace petals

	//spiral SDF
	/*
	color = vec3(1.);
	float spid = spiralD(p.xz, 1., 0.5, maturity);
	//float sphd = 0.01*length(p);
	//return max(spid, sphd);
	const float k = 0.005;
	float h = clamp(0.5 - 0.5*(spid-sphd)/k, 0.0, 1.0 );
	return mix( spid, sphd, h ) + k*h*(1.0-h);
	*/

	//float mouseFactor = 1.;
	//float mouseFactor = (1.-iMouse.y/iResolution.y);
	
	float noiseWarp = 0.3*(noise(7.*wp));
	float r1 = distance(vec2(-1., 0.), p.xy);
	float centerWarp = exp(-r1 * 2.);
	p.z -= centerWarp - 2.*max(0., 1. - centerWarp)*noiseWarp;
	
	p.xz += vec2(-0.2, 0.1)*exp2(-abs(p.y)); //middle ridge and pointy end
	//p.xz += clamp(2.*p.y, 0., 1.)*vec2(0., 0.1)*exp2(-abs(p.y)); //middle ridge and pointy end
	
	//p.x += abs(p.y); //beautiful star shaped
	//if(p.x < 0.)
	//	p.x -= 0.5*abs(p.y); //beautiful star shaped

	Hit hit;
	hit.d = diskD(p)*min3(scaling);
	hit.uv = diskP(p);
	hit.m = PETAL;

	return hit;
}


void opCheapBend(inout vec3 p, in float k)
{
	float c = cos(k*p.x);
	float s = sin(k*p.x);
	mat2  m = mat2(c,-s,s,c);
	vec3  q = vec3(m*p.xy,p.z);
	p = q;
}
//flower in the xy plane
const uint NB_PETALS = 5u;
Hit jasminD(in float r, in float maturity, in vec3 p, in vec3 wp)
{
	if(DEBUG_PETAL)
		return petal(r, maturity, rotateY(0.5*PI, p), wp, 0u);


	p = rotateY(0.5*PI, p);
	opCheapBend(p, 1.);
	p = rotateZ(0.23*PI, p);
	p = rotateY(-0.5*PI, p);

	Hit hit;
	hit.d = 2.*r;
	for(uint i = 0u; i < NB_PETALS; i++)
	{
		vec3 q = rotateZ((float(i)/float(NB_PETALS)*2. + 0.05)*PI, p);
		Hit newHit = petal(r, maturity, q, wp, i);
		if(newHit.d < hit.d)
			hit = newHit;
	}

	float sd = sphereD(vec3(0., 0., 0.28*r), 0.01, p);
	if(sd < hit.d)
	{
		hit.d = sd;
		hit.m = PISTIL;
	}

	//p = rotateY(0.5*PI, p);
	//p = rotateZ(-0.05*PI, p);
	//p = rotateY(-0.5*PI, p);

	//p = rotateY(0.5*PI, p);
	////p += 1.*r;
	////opCheapBend(p, 2.);
	////p -= 1.*r;
	//p = rotateY(-0.5*PI, p);

	sd = sdRoundCone(p, vec3(0., 0., 0.6*r), vec3(0., 0., 3.*r), 0.25*r, 0.2*r);
	if(sd < hit.d)
	{
		hit.d = sd;
		hit.m = STEM;
	}

	return hit;
}

//https://github.com/ssloy/tinyraytracer/wiki/Part-3:-shadertoy
#define JFIGW 32u
#define JFIGH 18u
uint[] jfig_bitfield = uint[](
	0x0u,0x0u,0x0u,0xf97800u,0x90900u,0xc91800u,0x890900u,0xf90900u,0x180u,
	//0x0u, 0x30e30e0u, 0x4904900u, 0x49e49e0u, 0x4824820u, 0x31e31e0u, 0x0u,0x0u,0x0u
	  0x0u, 0x40e30e0u, 0x4104900u, 0x41e49e0u, 0x4024820u, 0x41e31e0u, 0x0u,0x0u,0x0u
);
bool jfig(in vec2 uv)
{
	uvec2 ij = uvec2(uv * vec2(JFIGW, JFIGH) + (0.5, 0.));
	uint id = ij.x + (JFIGH-1u-ij.y)*JFIGW;
	if(id>=JFIGW*JFIGH) return false;
	return 0u != (jfig_bitfield[id/32u] & (1u << (id&31u)));
}

Hit map(in vec3 p)
{
	vec3 c = vec3(0.);
	p -= c;

	Hit hit;
	hit.d = 1000.;
	float d;

	//Table.
	d = diskD(rotateX(0.5*PI, p*3.) - vec3(0., 0., -0.35)) / 3.;
	if(d < hit.d)
	{
		hit.d = d;
		hit.m = TABLE;
	}

	//Flask.
	p = rotateY(-0.2*PI, p); //small rotation of the flask
	vec3 halfDiag = vec3(0.1, 0.1, 0.05);
	d = sdRoundBox(p, halfDiag, 0.01);
	if(d < hit.d)
	{
		hit.d = d;
		hit.m = FLASK;
	}

	//Label.
	float xmin = -0.8;
	float xmax =  0.8;
	float ymin = -0.8;
	float ymax =  0.5;
	float z = -1.25;
	vec3 pa = vec3(xmin, ymin, z) * halfDiag;
	vec3 pb = vec3(xmin, ymax, z) * halfDiag;
	vec3 pc = vec3(xmax, ymax, z) * halfDiag;
	vec3 pd = vec3(xmax, ymin, z) * halfDiag;
	d = udQuad(p, pa, pb, pc, pd);
	if(d < hit.d)
	{
		hit.d = d;
		hit.m = LABEL;
		hit.uv.xy = (p.xy/halfDiag.xy - vec2(xmin, ymin))/vec2(xmax - xmin, ymax - ymin);
		hit.uv.x = 1. - hit.uv.x;
	}

	//Cap.
	float h = 0.5*halfDiag.y;
	p.y -= halfDiag.y;
	d = sdCylinder(p, vec3(0), vec3(0., h, 0.), halfDiag.z);
	if(d < hit.d)
	{
		hit.d = d;
		hit.m = CAP;
		hit.uv = p;
	}

	//Flower
	//float radius = 0.15;
	float radius = 0.15;
	//float maturity = 2.*iMouse.x/iResolution.x;
	//float maturity = cos(iTime);
	float maturity = 1.3;
	//float maturity = 0.;
	//float maturity = cos(1. + 0.3*iTime);//2.*iMouse.x/iResolution.x;
	p.y += 2.*halfDiag.y;
	p.y -= radius/3. - 0.01;// + 0.025;
	p.x -= 2.*halfDiag.x;
	p.z += 1.5*halfDiag.z;
	p = rotateY(-2.1*PI, p);
	p = rotateZ(-0.1*PI, p);
	//p = rotateZ(2.*PI*iMouse.x/iResolution.x, p);
	p *= 3.;
	Hit jHit = jasminD(radius, maturity, p, p);
	jHit.d /= 3.;
	if(jHit.d < hit.d)
		return hit = jHit;

	return hit;
}

vec3 shade(in Hit hit)
{
	switch(hit.m)
	{
		case TABLE:
			return 2.*vec3(32.5, 15.6, 0.)/256.;
			//return vec3(1.);
			//return SUN_COLOR;
		case FLASK:
			return vec3(1.);
			break;
		case LABEL:
			//return vec3(0., 0., 1.);
			//return vec3(hit.uv.xy, 1.);
			return vec3(jfig(hit.uv.xy));
		case PISTIL:
			return vec3(1., 1., 0.);
		case PETAL:
			return vec3(1.);
			//return hit.uv.z <= 0. ? vec3(1.) : vec3(1., 0.2, 0.2);
		case CAP:
			return vec3(1.);
		case STEM:
			return vec3(1.);//10.*vec3(5.7, 19., 7.9)/256.;
		default:
			return vec3(0);
	}
}

vec3 randomDirection(in float u, in float v)
{
	float longitude = 2.*PI*u;
	float colatitude = acos(2.*v - 1.);
	float hRadius = sin(colatitude);
	return vec3(
		hRadius*cos(longitude),
		cos(colatitude),
		hRadius*sin(longitude)
	);
}
vec3 randomHemisphereDirection(in vec3 up, in float u, in float v)
{
	vec3 direction = randomDirection(u, v);
	return dot(direction, up) > 0. ? direction : -direction;
}
vec3 randomLambertianReflection(in vec3 normal, in float u, in float v)
{
	vec3 vec = (normal + randomDirection(u, v));
	//Avoiding degenerated case.
	return length(vec) < 0.001 ? normal : normalize(vec);
}


//----------------------------------------------------





//#define ZERO 0
#define ZERO (min(int(iTime),0))
#define ZEROu (uint(min(int(iTime),0)))
// http://iquilezles.org/www/articles/smin/smin.htm
vec3 smax( vec3 a, vec3 b, float k )
{
	vec3 h = max(k-abs(a-b),0.0);
	return max(a, b) + h*h*0.25/k;
}
vec3 sunlight(in vec3 d)
{
	if(!SUN) return vec3(0);
	return dot(-SUN_DIRECTION, d) > 0.99 ? SUN_COLOR*SUN_INTENSITY : vec3(0.);
}
vec3 background( in vec3 d, in bool fromSpecular )
{
	vec3 light = vec3(0.);

	//sky
	float t = 0.5*(d.y + 1.);
	vec3 skylight = (1. - t)*vec3(1.) + t*vec3(.5, .7, 1.);
	if(SKY)
		light += skylight;

	if(SUN)
		light += sunlight(d);

	return light;
}

// http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
vec3 calcNormal( in vec3 pos, in float eps )
{
	vec4 kk;
#if 0
	vec2 e = vec2(1.0,-1.0)*0.5773*eps;
	return normalize(
		e.xyy*map(pos + e.xyy).d +
		e.yyx*map(pos + e.yyx).d +
		e.yxy*map(pos + e.yxy).d +
		e.xxx*map(pos + e.xxx).d
	);
#else
	// inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
	vec3 n = vec3(0.0);
	for( int i=ZERO; i<4; i++ )
	{
		vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
		n += e*map(pos+eps*e).d;
	}
	return normalize(n);
#endif
}

const uint NB_STEPS = 128u;
const float EPSILON = 0.001;
const float MAX_DEPTH = 3.;
const float JACOBIAN_FACTOR = 1.5;
bool intersect(in vec3 ro, in vec3 rd, out Hit hit)
{
	float t = 0.;
	for(uint i = ZEROu; i < NB_STEPS && t < MAX_DEPTH; i++)
	{
		vec3 intersection = ro + t*rd;
		hit = map(intersection);
		hit.p = intersection;
		
		if(abs(hit.d) < EPSILON) return true;
		
		t += abs(hit.d)/JACOBIAN_FACTOR;
	}
	return false;
}

// Use Schlick's approximation for reflectance.
float reflectance(in float cosine, in float n1, in float n2)
{
	float r0 = sqr((n1 - n2)/(n1 + n2));
	return r0 + (1. - r0)*pow((1. - cosine), 5.);
}
float reflectance(float cosine, float ratio)
{
	return reflectance(cosine, 1., ratio);
}

void forcedScatter(inout vec3 rd, in Hit hit, inout vec3 attenuation, inout vec3 pdf, in vec3 dirOut)
{
	vec3 color = shade(hit);
	vec3 normal = hit.n;
	switch(hit.m)
	{
		case TABLE:
		case LABEL:
		case PISTIL:
		case PETAL:
		{
			float p = dot(normal, dirOut)/PI;
			float d = dot(normal, dirOut);
			attenuation *= 1./PI * d * color;
			rd = dirOut;
			pdf *= p;
			return;
		}
		case FLASK:
		case CAP:
		{
			attenuation = vec3(0.);
			pdf = vec3(0.);
			return;
		}
	}
}
bool isDelta(in uint m)
{
	switch(m)
	{
		case TABLE:
		case LABEL:
		case PISTIL:
		case PETAL:
		case STEM:
			return false;
		case FLASK:
		case CAP:
			return true;
	}
}
bool scatter(inout vec3 ro, inout vec3 rd, in Hit hit, inout vec3 attenuation, inout vec3 pdf)
{
	vec3 color = shade(hit);
	vec3 normal = hit.n;
	switch(hit.m)
	{
		case TABLE:
		case LABEL:
		case PISTIL:
		case PETAL:
		case STEM:
		{
			vec3 bounceDirection = randomLambertianReflection(normal, randUniform(), randUniform());
			float p = dot(normal, bounceDirection)/PI;
			float d = dot(normal, bounceDirection);
			attenuation *= 1./PI * d * color;
			ro = hit.p + normal*2.*EPSILON;
			rd = bounceDirection;
			pdf *= p;
			return true;
		}
		//mirror
		//{
		//	vec3 reflectDirection = reflect(rd, normal);
		//	attenuation *= color;
		//	ro = hit.p + normal*2.*EPSILON;
		//	rd = reflectDirection;
		//	return true;
		//}
		case FLASK:
		case CAP:
		{
			const float airRefractionIndex = 1.;
			const float glassRefractionIndex = 1.5;
			float refractionIndexRatio = airRefractionIndex/glassRefractionIndex;
			
			bool backFace = dot(rd, normal) > 0.;
			//bool backFace = hit.d < 0.;
			if(backFace) //swap for backface
			{
				refractionIndexRatio = 1./refractionIndexRatio;
				normal = -normal;
			}
			
			float cosTheta = dot(-rd, normal);
			float sinTheta = sqrt(1. - sqr(cosTheta));
			float reflectedRatio = sinTheta * refractionIndexRatio > 1. ? 1. : reflectance(cosTheta, refractionIndexRatio);
			
			//Total internal reflection, or both reflection and refraction but with Schlik's approximation giving the reflectance (that cancels out by stochastic selection).
			if(reflectedRatio == 1. || reflectedRatio > randUniform())
			{
				attenuation *= reflectedRatio;
				pdf *= reflectedRatio;
				rd = reflect(rd, normal);
			}
			else
			{
				attenuation *= (1. - reflectedRatio);
				pdf *= (1. - reflectedRatio);
				rd = refract(rd, normal, refractionIndexRatio);
			}
			
			ro = hit.p - normal*2.*EPSILON;
			attenuation *= color;
			return true;
		}
	}
	return false;
}

//----------------------------------------------------

mat3 observerViewMatrix(in vec2 mouseUV)
{
	vec2 shift = vec2(0.);
	float a = (shift.x + mouseUV.x)*PI*2.;
	float b = (shift.y + mouseUV.y)*PI;
	
	vec3 camera = vec3(cos(b)*sin(a), sin(b), cos(b)*cos(a));
	//Z vector
	vec3 up = normalize(cross(cross(camera, vec3(0, 1, 0)), camera));
	//Y vector
	vec3 x = normalize(cross(up, camera));
	
	const float depth = 0.75;
	return depth*mat3(x, up, camera);
}
void pixelRay(in vec2 ij, out vec3 ro, out vec3 rd)
{
	//Towards -Z.
	ro = vec3(0, 0, 1);
	vec2 q = (ij - 0.5*iResolution.xy)/iResolution.y;
	rd = normalize(vec3(q, 0) - ro);

	mat3 view = observerViewMatrix(CONTROL_CAMERA ? iMouse.xy/iResolution.xy - vec2(0., 0.5) : vec2(0.5, 0.1));
	ro = view[2];
	rd = normalize(view*rd);
}

void debugSDF(out vec4 fragColor, in vec2 fragCoord)
{
	//Point p in [-1, 1]x[-1, 1] of xy plan.
	vec3 p = vec3(fragCoord.xy/iResolution.xy - 0.5, 0.);
	p = rotateX(0.5*PI, p);
	p += vec3(0., 0., -0.1);

	//float d = sphereD(p);
	float d = map(p).d;

	//Coloring
	//https://www.shadertoy.com/view/3ltSW2
	vec3 color = vec3(1.0) - sign(d)*vec3(0.1,0.4,0.7);
	color *= 1.0 - exp(-3.0*abs(d));
	color *= 0.8 + 0.2*cos(150.0*d);
	color = mix( color, vec3(1.0), 1.0-smoothstep(0.0,0.01,abs(d)) );

	fragColor = vec4(color, 1.);
}

#define SAMPLES 4u
#define BOUNCES 50u
void render(out vec4 fragColor, in vec2 fragCoord)
{
	if(DEBUG_PETAL)
	{
		debugSDF(fragColor, fragCoord);
		return;
	}

	fragColor = vec4(vec3(0.), 1.);

	vec3 ro;
	vec3 rd;
	pixelRay(fragCoord.xy + vec2(randUniform(), randUniform()), ro, rd);
	vec3 attenuation = vec3(1.);
	vec3 pdf = vec3(1.);
	bool delta = true;

	for(uint b = BOUNCES; b > ZEROu; b--)
	{
		Hit hit;
		if(!intersect(ro, rd, hit))
		{
			fragColor.xyz += attenuation/pdf * background(rd, delta);
			break;
		}
		
		hit.n = calcNormal(hit.p - rd*2.*EPSILON, 0.0001);
		delta = isDelta(hit.m);
		
		//NEE
		if(NEE && !delta)
		{
			vec3 neAttenuation = attenuation;
			vec3 nePdf = pdf*0.5/PI;
			vec3 outDir = randomHemisphereDirection(hit.n, randUniform(), randUniform());
			
			Hit neHit = hit;
			neHit.p += normalize(hit.n - rd)*1.*EPSILON;
			//neHit.p += hit.n*2.*EPSILON;
			//neHit.p += -outDir*2.*EPSILON;
			
			forcedScatter(rd, neHit, neAttenuation, nePdf, outDir);
			
			if(!intersect(neHit.p, -outDir, neHit))
				fragColor.xyz += neAttenuation/nePdf * background(outDir, delta);
		}
		
		if(!scatter(ro, rd, hit, attenuation, pdf))
			break;
	}
}




void main(/*out vec4 fragColor, in vec2 fragCoordv*/)
{
	iMouse = vec4(mouse.xyzz);
	iResolution = vec2(imageSize(image));
	vec2 pixel = vec2(gl_GlobalInvocationID.xy);
	if(any(greaterThanEqual(pixel, iResolution))) return;

	vec4 oldColor = imageLoad(image, ivec2(pixel));
	for(uint i = ZEROu; i < SAMPLES; i++)
	{
		initRandState(iTime, uvec2(pixel), uvec2(iResolution.xy), uint(oldColor.w));
		
		vec4 newColor;
		render(newColor, pixel);

		//Avoid degenerate paths.
		if(any(isnan(newColor)) || any(lessThan(newColor, vec4(0.))))
			continue;

		if(iMouse.z != 0.)
		{
			fragment_color = newColor;
			imageStore(image, ivec2(pixel), fragment_color);
			return;
		}

		float count = oldColor.a + newColor.a;
		oldColor = vec4(oldColor.rgb + (newColor.rgb - oldColor.rgb) / count, count);
	}

	
	fragment_color = oldColor;
	imageStore(image, ivec2(pixel), fragment_color);
}
///////
#endif
