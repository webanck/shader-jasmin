vec2 cartesianToPolar(in vec2 p)
{
	float r = length(p);
	float theta = atan(p.y, p.x);
	return vec2(r, theta);
}


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

//https://www.shadertoy.com/view/7lS3Dw
//3d generalization of https://www.shadertoy.com/view/ttlGDf
vec3 warp3d(vec3 pos, float t) {
	float curv =.35, a = 1.2, b = 0.2;
	pos *= 3.;
	for(float k = 1.0; k < 4.0; k += 1.0)
	{
		pos.x += curv * sin(t + k * a * pos.y) + t * b;
		pos.y += curv * cos(t + k * a * pos.x);
		pos.y += curv * sin(t + k * a * pos.z) + t * b;
		pos.z += curv * cos(t + k * a * pos.y);
		pos.z += curv * sin(t + k * a * pos.x) + t * b;
		pos.x += curv * cos(t + k * a * pos.z);
	}
	return 0.5 + 0.5 * cos(pos.xyz + vec3(1,2,4));
}





const float PI = 3.14;


vec3 rotateZ(in float rads, in vec3 p)
{
	return vec3(
		cos(rads)*p.x - sin(rads)*p.y,
		sin(rads)*p.x + cos(rads)*p.y,
		p.z
	);
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
float bentDisk(in vec3 p)
{
    const float k = 20.*iMouse.x/iResolution.x; // or some other amount
    float c = cos(k*p.y);
    float s = sin(k*p.y);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xy,p.z);
    return diskD(q);
	////return sdBox(p, vec3(1.)) - 0.9*exp(-distance(p, vec3(0., 0., 0.)));
	//return diskD(p) - 0.3*exp(-distance(p, vec3(0., 0., 0.5)));
	//return diskD(p - 0.3*(p - vec3(0, 0.5, 0.2)));
	//return diskD(vec3(p.xy, p.z + k * p.x*p.x));
}
void spiralify(inout vec2 p)
{
	vec2 mouseFactors = iMouse.xy/iResolution.xy;
	mouseFactors.y = 1. - mouseFactors.y;

	float cst = 6.*mouseFactors.y;
	const float a = 3.;
	const float b = 3.;
	//const float a = 10. * mouseFactors.x;
	//const float b = 10. * mouseFactors.y;
	float r = length(p);//sqrt(x*x + y*y);
	float angle = acos(p.x/r) * (p.y < 0.0 ? -1.0:1.0);
	float spiralangle = angle + b * PI * log(r * a + cst);
	//float spiralangle = angle + b * PI * log(pow(r, 5.*mouseFactors.x) * a + cst - 3.);
	//float spiralangle = angle + b * PI * log(r * a + cst) * clamp(1. - 13.3*r, 0., 1.);
	//TODO: find why 13.3 ?

	//p = r * vec2(cos(spiralangle), sin(spiralangle));
	//p = mix(r * vec2(cos(spiralangle), sin(spiralangle)), p, clamp(mouseFactors.x * 15.*r, 0., 1.));
	p = mix(r * vec2(cos(spiralangle), sin(spiralangle)), p, clamp(13.3*r, 0., 1.));
}


//----------------------------------------------------

//maturity in [0, 2]
float petal(in float r, in float maturity, in vec3 p, in vec3 wp)
{
	//a sphere describing the extremity of the petal trajectory (rotation) through maturation
	//vec3 c = vec3(0, 0, r);
	//return sphereD(c, 0.1*r, rotateY(0.5*maturity*PI, p));

	//TODO: how to create creases/ridges in the middle of the petal as this is doing at a certain angle ?
	//p.z += 0.7*abs(noise(p));

	//Spiral at the end of the petal.
	p.xz -= vec2(0.3, -0.);
	spiralify(p.xz);
	p.xz += vec2(0.3, -0.);

	//Big spiral at the origin of the petal.
	p.xz /= 3.;
	spiralify(p.xz);
	p.xz *= 3.;

	//scale
	p /= r*vec3(1, 0.5, 1.);
	//rotate
	//p = rotateY(0.5*maturity*PI, p);
	//shift
	p.x -= 1.;

	float mouseFactor = 1.;
	//float mouseFactor = (1.-iMouse.y/iResolution.y);
	//p += warp3d(wp, 0.3*iTime);
	p.z += 2.*mouseFactor * 0.3*(noise(7.*wp));
	//p.x += abs(p.y); //beautiful star shaped
	p.z += mouseFactor * 0.1*exp2(-abs(p.y)); //middle ridge
	//p += 0.1*warp3d(p*0.3, 0.);

	return diskD(p);
}

//flower in the xy plane
const uint NB_PETALS = 5u;
float jasmin(in float r, in float maturity, in vec3 p, in vec3 wp)
{
	//return petal(r, maturity, rotateY(0.5*PI, p), wp);

	float d = 2.*r;
	for(uint i = 0u; i < NB_PETALS; i++)
		d = min(d, petal(r, maturity, rotateZ(float(i)/float(NB_PETALS)*2.*PI, p), wp));
	return d;
}

float jasminD(in vec3 c, in float r, in vec3 orientation, in float maturity, in vec3 p)
{
	return jasmin(r, maturity, p - c, p); //todo: p orientation transform
}

float map(in vec3 p)
{
	float radius = 0.15;
	//float maturity = 2.*iMouse.x/iResolution.x;
	//float maturity = 0.3;
	float maturity = 0.;
	//float maturity = cos(1. + 0.3*iTime);//2.*iMouse.x/iResolution.x;
	return jasminD(vec3(0, 0, -0.1), radius, vec3(0, 0, 1), maturity, p);
}

//----------------------------------------------------

// http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
//#define ZERO 0
#define ZERO (min(int(iTime),0))
vec3 calcNormal( in vec3 pos, in float eps )
{
	vec4 kk;
#if 0
	vec2 e = vec2(1.0,-1.0)*0.5773*eps;
	return normalize(
		e.xyy*map(pos + e.xyy) +
		e.yyx*map(pos + e.yyx) +
		e.yxy*map(pos + e.yxy) +
		e.xxx*map(pos + e.xxx)
	);
#else
	// inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
	vec3 n = vec3(0.0);
	for( int i=ZERO; i<4; i++ )
	{
		vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
		n += e*map(pos+eps*e);
	}
	return normalize(n);
#endif
}

const float STEP_SIZE = 0.01;
const float NB_STEPS = 200u;
const float EPSILON = 1.1*STEP_SIZE;
const float MAX_DEPTH = 10.;
const float JACOBIAN_FACTOR = 20.;
bool intersect(in vec3 ro, in vec3 rd, out vec3 intersection)
{
	float t = 0.;
	for(uint i = 0u; i < NB_STEPS && t < MAX_DEPTH; i++)
	{
		intersection = ro + t*rd;
		float d = map(intersection);
		if(d < EPSILON) return true;
		
		t += max(STEP_SIZE, d)/JACOBIAN_FACTOR;
	}
	return false;
}

//TODO: softshadow

//----------------------------------------------------

float obsDistance(const uint obsId)
{
	//Inside view.
	if(obsId == 0u)
		return 1.;
	//Outside view.
	else
		return 8.;
}
mat3 observerViewMatrix(const uint obsId, vec2 mouseUV)
{
	vec2 shift = vec2(0.);
	float a = (shift.x + mouseUV.x)*PI*2.;
	float b = (shift.y + mouseUV.y)*PI;
	//if (mouseUV == vec2(0)) {b = -0.2; a = 0.2;}
	//if(obsId == 1u)
	//{
	//	a *= -1.;
	//	b *= -1.;
	//}
	
	vec3 camera = vec3(cos(b)*sin(a), sin(b), cos(b)*cos(a));
	//Z vector
	vec3 up = normalize(cross(cross(camera, vec3(0, 1, 0)), camera));
	//Y vector
	vec3 x = normalize(cross(up, camera));
	
	//inside view
	return obsDistance(obsId)*mat3(x, up, camera);
}
void pixelRay(in vec2 ij, out vec3 ro, out vec3 rd)
{
	//Towards -Z.
	ro = vec3(0, 0, 1);
	vec2 q = (ij - 0.5*iResolution.xy)/iResolution.y;
	rd = normalize(vec3(q, 0) - ro);

	mat3 view = observerViewMatrix(0u, iMouse.xy/iResolution.xy - 0.5);
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
	//float d = bentBox(p, vec3(0.1));
	float d = map(p);

	//Coloring
	//const float dMin = 0.;
	//const float dMax = 2.;
	//d = (clamp(d, dMin, dMax) - dMin)/(dMax - dMin);
	//vec3 color = vec3(d);
	//
	////https://www.shadertoy.com/view/3t33WH
	//vec3 color = (d<0.0) ? vec3(0.6,0.8,1.0) : vec3(0.9,0.6,0.3);
	//color *= 1.0 - exp(-9.0*abs(d));
	//color *= 1.0 + 0.2*cos(128.0*abs(d));
	//color = mix( color, vec3(1.0), 1.0-smoothstep(0.0,0.015,abs(d)) );
	//
	//https://www.shadertoy.com/view/3ltSW2
	vec3 color = vec3(1.0) - sign(d)*vec3(0.1,0.4,0.7);
	color *= 1.0 - exp(-3.0*abs(d));
	color *= 0.8 + 0.2*cos(150.0*d);
	color = mix( color, vec3(1.0), 1.0-smoothstep(0.0,0.01,abs(d)) );

	fragColor = vec4(color, 1.);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
	//debugSDF(fragColor, fragCoord); return;

	vec3 ro;
	vec3 rd;
	pixelRay(fragCoord.xy, ro, rd);
	vec3 intersection;
	bool intersects = intersect(ro, rd, intersection);
	if(intersects)
	{
		vec3 normal = calcNormal(intersection, 0.0001);
		vec3 color = 0.5 + 0.5*normal;
		vec3 lightDir =
			vec3(0., 0., 1.);
			//normalize(vec3(1.));
		fragColor = vec4(color * abs(dot(rd, normal)), 1.);
	}
	else fragColor = vec4(vec3(0.), 1.);
}
