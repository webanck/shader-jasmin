#define DEBUG_PETAL false


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
};

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
float sdRoundedCylinder( vec3 p, float ra, float rb, float h )
{
  vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}
float dot2( in vec3 v ) { return dot(v,v); }
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
	//return sdRoundedCylinder(p.xzy, 1., 0.05, 0.01);

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

struct Hit
{
	float d;
	vec3 uv;
	vec3 p;
	uint m;
};

//maturity in [0, 2]
Hit petal(in float r, in float maturity, in vec3 p, in vec3 wp)
{
	//TODO: how to create creases/ridges in the middle of the petal as this is doing at a certain angle ?
	//p.z += 0.7*abs(noise(p));

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

	float mouseFactor = 1.;
	//float mouseFactor = (1.-iMouse.y/iResolution.y);
	p.z += 2.*mouseFactor * 0.3*(noise(7.*wp));
	//p.x += abs(p.y); //beautiful star shaped
	p.z += mouseFactor * 0.1*exp2(-abs(p.y)); //middle ridge

	Hit hit;
	hit.d = diskD(p)*min3(scaling);
	hit.uv = diskP(p);
	hit.m = PETAL;
	return hit;
}

//flower in the xy plane
const uint NB_PETALS = 5u;
Hit jasmin(in float r, in float maturity, in vec3 p, in vec3 wp)
{
	if(DEBUG_PETAL)
		return petal(r, maturity, rotateY(0.5*PI, p), wp);

	Hit hit;
	hit.d = 2.*r;
	for(uint i = 0u; i < NB_PETALS; i++)
	{
		Hit newHit = petal(r, maturity, rotateZ(float(i)/float(NB_PETALS)*2.*PI, p), wp);
		if(newHit.d < hit.d)
			hit = newHit;
	}

	float sd = sphereD(vec3(0), 0.01, p);
	if(sd < hit.d)
	{
		hit.d = sd;
		hit.m = PISTIL;
	}

	return hit;
}

Hit jasminD(in vec3 c, in float r, in vec3 orientation, in float maturity, in vec3 p)
{
	return jasmin(r, maturity, p - c, p); //todo: p orientation transform
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

Hit flaskD(in vec3 c, in vec3 p)
{
	p -= c;

	Hit hit;
	hit.d = 1000.;
	float d;

	//Table.
	d = diskD(rotateX(0.5*PI, p*3.)-vec3(0., 0., -0.35)) / 3.;
	if(d < hit.d)
	{
		hit.d = d;
		hit.m = TABLE;
	}

	//Flask.
	p = rotateY(-0.2*PI, p); //small rotation of the flask
	vec3 diag = vec3(0.1, 0.1, 0.05);
	d = sdRoundBox(p, diag, 0.01);
	if(d < hit.d)
	{
		hit.d = d;
		hit.m = FLASK;
	}

	//Label.
	//p -= -1.5*diag.z;
	float xmin = -0.8;
	float xmax =  0.8;
	float ymin = -0.8;
	float ymax =  0.5;
	float z = -1.25;
	vec3 pa = vec3(xmin, ymin, z) * diag;
	vec3 pb = vec3(xmin, ymax, z) * diag;
	vec3 pc = vec3(xmax, ymax, z) * diag;
	vec3 pd = vec3(xmax, ymin, z) * diag;
	d = udQuad(p, pa, pb, pc, pd);
	if(d < hit.d)
	{
		hit.d = d;
		hit.m = LABEL;
		hit.uv.xy = (p.xy/diag.xy - vec2(xmin, ymin))/vec2(xmax - xmin, ymax - ymin);
		hit.uv.x = 1. - hit.uv.x;
	}

	return hit;
}

Hit map(in vec3 p)
{
	vec3 c = vec3(0.);

	//Flask and table.
	Hit fHit = flaskD(c, p);

	//float radius = 0.15;
	float radius = 0.15;
	//float maturity = 2.*iMouse.x/iResolution.x;
	//float maturity = cos(iTime);
	float maturity = 1.3;
	//float maturity = 0.;
	//float maturity = cos(1. + 0.3*iTime);//2.*iMouse.x/iResolution.x;
	c = vec3(-0.2, 0.2, 0.)*3.;
	Hit jHit = jasminD(c, radius, vec3(0, 0, 1), maturity, p*3.);
	jHit.d /= 3.;

	if(fHit.d < jHit.d)
		return fHit;
	return jHit;
}
vec3 shade(in Hit hit)
{
	vec3 color = vec3(0.);
	switch(hit.m)
	{
		case TABLE:
			color = 2.*vec3(32.5, 15.6, 0.)/256.;
			break;
		case FLASK:
			color = vec3(0.5);
			break;
		case LABEL:
			//color = vec3(0., 0., 1.);
			//color = vec3(hit.uv.xy, 1.);
			color = vec3(jfig(hit.uv.xy));
			break;
		case PISTIL:
			color = vec3(1., 1., 0.);
			break;
		case PETAL:
			color = hit.uv.z <= 0. ? vec3(1.) : vec3(1., 0.2, 0.2);
			break;
	}
	return color;
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
vec3 background( in vec3 d )
{
	return vec3(0.2, 0.8, 0.2);

    // cheap cubemap
    vec3 n = abs(d);
    vec2 uv = (n.x>n.y && n.x>n.z) ? d.yz/d.x: 
              (n.y>n.x && n.y>n.z) ? d.zx/d.y:
                                     d.xy/d.z;
    // fancy blur
    vec3  col = vec3( 0.0 );
    for( int i=ZERO; i<200; i++ )
    {
        float h = float(i)/200.0;
        float an = 31.0*6.2831*h;
        vec2  of = vec2( cos(an), sin(an) ) * h;

        vec3 tmp = vec3(noise(vec3(uv*0.25 + 0.0075*of, 0.)));//;texture( iChannel2, uv*0.25 + 0.0075*of, 4.0 ).yxz;
        col = smax( col, tmp, 0.5 );
    }
    
    return pow(col,vec3(3.5,3.0,6.0))*0.2;
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
const float JACOBIAN_FACTOR = 1.1;
bool intersect(in vec3 ro, in vec3 rd, out Hit hit)
{
	float t = 0.;
	for(uint i = ZEROu; i < NB_STEPS && t < MAX_DEPTH; i++)
	{
		vec3 intersection = ro + t*rd;
		hit = map(intersection);
		hit.p = intersection;
		
		if(hit.d < EPSILON) return true;
		
		t += hit.d/JACOBIAN_FACTOR;
	}
	return false;
}

//TODO: softshadow
//TODO: refraction/reflection
// Use Schlick's approximation for reflectance.
//float reflectance(in float cosine, in float n1, in float n2)
//{
//	float r0 = sqr((n1 - n2)/(n1 + n2));
//	return r0 + (1. - r0)*pow((1. - cosine), 5.);
//}

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
	
	const float depth = 1.;
	return depth*mat3(x, up, camera);
}
void pixelRay(in vec2 ij, out vec3 ro, out vec3 rd)
{
	//Towards -Z.
	ro = vec3(0, 0, 1);
	vec2 q = (ij - 0.5*iResolution.xy)/iResolution.y;
	rd = normalize(vec3(q, 0) - ro);

	mat3 view = observerViewMatrix(iMouse.xy/iResolution.xy - vec2(0., 0.5));
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
#define RENDER_AA 2u
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
	if(DEBUG_PETAL)
	{
		debugSDF(fragColor, fragCoord);
		return;
	}

	fragColor = vec4(vec3(0.), 1.);
	for(uint i = 0u; i < RENDER_AA; i++)
	for(uint j = 0u; j < RENDER_AA; j++)
	{
		vec3 ro;
		vec3 rd;
		pixelRay(fragCoord.xy + (vec2(i, j) + 0.5)/vec2(RENDER_AA), ro, rd);
		Hit hit;
		
		bool intersects = intersect(ro, rd, hit);
		if(intersects)
		{
			vec3 normal = calcNormal(hit.p, 0.0001);
			vec3 color = shade(hit);
			//color = 0.5 + 0.5*normal;
			vec3 lightDir =
				//vec3(0., 0., 1.);
				normalize(vec3(1., -1., 1.));
			//check visibility/shadowing
			intersects = intersect(hit.p + normal*2.*EPSILON, -lightDir, hit);
			float visibility = intersects ? 0. : 1.;
			vec3 lightIntensity = vec3(1.);
			fragColor.xyz += color * abs(dot(rd, normal)) * (0.5 + visibility * max(0., dot(-lightDir, normal))) * lightIntensity;
		}
		else fragColor.xyz += background(rd);
	}
	fragColor.xyz /= float(RENDER_AA*RENDER_AA);
}
