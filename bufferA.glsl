#define CONTROL_CAMERA true
#define SKY true
#define SUN true
#define SAMPLES_PER_FRAME 4u
#define BOUNCES 10u
#define BIFACE true

#define NO_MIS 0u
#define BALANCED_MIS 1u
#define SMART_MIS 2u
const uint MIS_TYPE = BALANCED_MIS;

const float PI = 3.14;
#define SUN_DIRECTION normalize(vec3(-1.))
//#define SUN_DIRECTION vec3(0., -1., 0.)
const float SUN_INTENSITY = 314.;
#define SUN_COLOR vec3(249, 231, 42)/256.
const float SUN_APPARENT_RATIO = 0.02;

//Raymarching intersections.
const uint NB_STEPS = 128u;
const float EPSILON = 0.001;
const float MAX_DEPTH = 1.;
const float JACOBIAN_FACTOR = 1.1;

//#define ZERO 0
#define ZERO (min(int(iTime),0))
#define ZEROu (uint(min(int(iTime),0)))


float dot2(vec3 v) { return dot(v,v); }
float dot2(vec2 v) { return dot(v,v); }
float min3(vec3 v) { return min(min(v.x, v.y), v.z); }
float sqr(float x) { return x*x; }

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

//-----------------------
const bool CORRELATED_SAMPLES = false;
//https://www.pcg-random.org/
uint pcg(inout uint state)
{
	state = state * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}
uint randState = 0u;
void initRandState(const float time, const uvec2 pixel, const uvec2 resolution, const uint iteration)
{
	uint i = iteration;
	if(CORRELATED_SAMPLES)
		randState = i;
	else
		randState = pixel.x + resolution.x*(pixel.y + resolution.y*i);
}
float randUniform()
{
	return fract(float(pcg(randState)) / 4294967296.);
}

//-----------------------


//https://www.shadertoy.com/view/4djSRW
vec3 hash(vec3 p3)
{
	p3 = fract(p3 * vec3(.1031, .1030, .0973));
	p3 += dot(p3, p3.yxz+33.33);
	return fract((p3.xxy + p3.yxx)*p3.zyx);
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
	float baba = dot2(ba);
	float paba = dot(pa, ba);
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
float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2)
{
	// sampling independent computations (only depend on shape)
	vec3  ba = b - a;
	float l2 = dot2(ba);
	float rr = r1 - r2;
	float a2 = l2 - rr*rr;
	float il2 = 1.0/l2;

	// sampling dependant computations
	vec3 pa = p - a;
	float y = dot(pa, ba);
	float z = y - l2;
	float x2 = dot2( pa*l2 - ba*y );
	float y2 = y*y*l2;
	float z2 = z*z*l2;

	// single square root!
	float k = sign(rr)*rr*rr*x2;
	if( sign(z)*a2*z2 > k ) return sqrt(x2 + z2)*il2 - r2;
	if( sign(y)*a2*y2 < k ) return sqrt(x2 + y2)*il2 - r1;
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
		dot(nor,pa)*dot(nor,pa)/dot2(nor)
	);
}
void opCheapBend(inout vec3 p, in float k)
{
	float c = cos(k*p.x);
	float s = sin(k*p.x);
	mat2  m = mat2(c,-s,s,c);
	p = vec3(m*p.xy,p.z);
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
	float r2 = dot2(p.xy);
	//Projection in the disk, closest is disk surface.
	if(r2 < 1.0) return abs(p.z);
	
	//Projection out of disk, closest is distance to circle.
	float r = sqrt(r2);
	float h = r - 1.0;
	return sqrt(h*h + sqr(p.z));
}
vec3 diskP(in vec3 p)
{
	float r2 = dot2(p.xy);
	//Projection in the disk, closest is disk surface.
	if(r2 < 1.0) return vec3(p.xy, sign(p.z));
	
	//Projection out of disk, closest is distance to circle.
	float r = sqrt(r2);
	return vec3(cos(r), sin(r), sign(p.z));
}

//----------------------------------------------------
#define TABLE 0u
#define FLASK 1u
#define LABEL 2u
#define PISTIL 3u
#define PETAL 4u
#define STEM 6u
#define CAP 8u

struct Hit
{
	float d;
	vec3 uv;
	vec3 p;
	uint m;
	vec3 n;
};

Hit petal(in float r, in vec3 p, in vec3 wp, in uint i)
{
	//scale
	vec3 scaling = r*vec3(1, 0.5, 1.);
	p /= scaling;
	//shift
	p.x -= 1.;

	float noiseWarp = 0.3*(noise(vec3(7.*wp.xy, 0.)));
	float r1 = distance(vec2(-1., 0.), p.xy);
	float centerWarp = exp(-r1 * 2.);
	p.z -= centerWarp - 2.*max(0., 1. - centerWarp)*noiseWarp;
	
	p.xz += vec2(-0.2, 0.1)*exp2(-abs(p.y)); //middle ridge and pointy end

	Hit hit;
	hit.d = diskD(p)*min3(scaling);
	hit.uv = diskP(p);
	hit.m = PETAL;

	return hit;
}


//flower in the xy plane
const uint NB_PETALS = 5u;
Hit jasminD(in float r, in vec3 p, in vec3 wp)
{
	p = rotateY(0.5*PI, p);
	opCheapBend(p, 1.);
	p = rotateZ(0.23*PI, p);
	p = rotateY(-0.5*PI, p);

	Hit hit;
	hit.d = 2.*r;
	for(uint i = 0u; i < NB_PETALS; i++)
	{
		vec3 q = rotateZ(2.*PI*float(i)/float(NB_PETALS), p);
		Hit newHit = petal(r, q, p, i);
		if(newHit.d < hit.d)
			hit = newHit;
	}

	float sd = sphereD(vec3(0., 0., 0.28*r), 0.01, p);
	if(sd < hit.d)
	{
		hit.d = sd;
		hit.m = PISTIL;
	}

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
	//2020
	//0x0u, 0x30e30e0u, 0x4904900u, 0x49e49e0u, 0x4824820u, 0x31e31e0u, 0x0u,0x0u,0x0u
	//hello 2021!
	  0x0u, 0x40e30e0u, 0x4104900u, 0x41e49e0u, 0x4024820u, 0x41e31e0u, 0x0u,0x0u,0x0u
);
bool jfig(in vec2 uv)
{
	uvec2 ij = uvec2(uv * vec2(JFIGW, JFIGH));
	uint id = ij.x + (JFIGH-1u-ij.y)*JFIGW;
	if(id>=JFIGW*JFIGH) return false;
	return 0u != (jfig_bitfield[id/32u] & (1u << (id&31u)));
}

Hit map(in vec3 p)
{
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
	float radius = 0.15;
	p.y += 2.*halfDiag.y;
	p.y -= radius/3. - 0.009;
	p.x -= 2.*halfDiag.x;
	p.z += 1.5*halfDiag.z;
	p = rotateY(-2.1*PI, p);
	p = rotateZ(-0.04*PI, p);
	p *= 3.;
	Hit jHit = jasminD(radius, p, p);
	jHit.d /= 3.;
	if(jHit.d < hit.d)
		hit = jHit;

	return hit;
}

vec3 shade(in Hit hit)
{
	switch(hit.m)
	{
		case TABLE:
			return 2.*vec3(32.5, 15.6, 0.)/256.;
		case LABEL:
			return vec3(jfig(hit.uv.xy));
		case PISTIL:
			return vec3(1., 1., 0.);
		case FLASK:
		case PETAL:
		case CAP:
		case STEM:
			return vec3(1.);
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
float sphereCapArea(in float angle)
{
	return 2.*PI * (1. - cos(angle));
}
//sample sphere cap of given angle at top of z axis
vec3 randomSphereCapDirection(in float u, in float v, in float angle, out float pdf)
{
	float longitude = 2.*PI*u;
	float colatitude = acos(1. - v*(1. - cos(angle)));
	float hRadius = sin(colatitude);
	
	pdf = 1./sphereCapArea(angle);
	
	return vec3(
		hRadius*cos(longitude),
		hRadius*sin(longitude),
		cos(colatitude)
	);
}
//sample sphere cap at any axis
vec3 randomSphereCapDirection(in float u, in float v, in vec3 direction, in float angle, out float pdf)
{
	float longitude = atan(direction.y, direction.x);
	float colatitude = acos(direction.z);
	
	//3 vectors forming the base (dtheta, dphi, dr)
	float cc = cos(colatitude);
	float sc = sin(colatitude);
	float cl = cos(longitude);
	float sl = sin(longitude);
	vec3 dtheta = vec3(
		-sl,
		cl,
		0
	);
	vec3 dphi = vec3(
		cc*cl,
		cc*sl,
		-sc
	);
	vec3 dr = vec3(
		sc*cl,
		sc*sl,
		cc
	);
	//define the transformation matrix
	mat3 m = mat3(dtheta, dphi, dr);
	
	return m * randomSphereCapDirection(u, v, angle, pdf);
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




bool isSunDirection(in vec3 direction)
{
	return dot(-SUN_DIRECTION, direction) > (1. - 0.5*SUN_APPARENT_RATIO);
}
vec3 sunlight(in vec3 direction)
{
	return isSunDirection(direction) ? SUN_COLOR*SUN_INTENSITY : vec3(0.);
}
vec3 background(in vec3 d)
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
	//0.05*PI is the smallest angle without loss of energy
float randomSunlightPdf(in vec3 direction)
{
	return isSunDirection(direction) ? 1./sphereCapArea(max(0.05, SUN_APPARENT_RATIO*0.5)*PI) : 0.;
}
vec3 randomSunlightDirection(in float u, in float v, out float pdf, in vec3 n)
{
	return randomSphereCapDirection(u, v, -SUN_DIRECTION, max(0.05, SUN_APPARENT_RATIO*0.5)*PI, pdf);
}

// http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
vec3 calcNormal( in vec3 pos, in float eps )
{
	vec4 kk;
	// inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
	vec3 n = vec3(0.0);
	for( int i=ZERO; i<4; i++ )
	{
		vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
		n += e*map(pos+eps*e).d;
	}
	return normalize(n);
}

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

bool scatter(inout vec3 ro, inout vec3 rd, in Hit hit, inout vec3 attenuation, inout vec3 pdf)
{
	if(dot2(hit.n) < 0.5) return false;
	vec3 color = shade(hit);
	switch(hit.m)
	{
		//Diffuse materials.
		case PISTIL:
		case PETAL:
		case STEM: if(BIFACE)
		{
			bool backFace = dot(rd, hit.n) > 0.;
			//bool backFace = hit.d < 0.;
			if(backFace) //swap for backface
			{
				hit.n = -hit.n;
			}
			const float transmittance = 0.1;
			float pPassThrough = transmittance/(transmittance + 1.);
			if(randUniform() < pPassThrough)
			{
				hit.n = -hit.n;
				while(abs(map(hit.p).d) < EPSILON)
					hit.p += hit.n*EPSILON;
				//hit.p += hit.n*5.*EPSILON;
				Hit newHit = map(hit.p);
				if(abs(newHit.d) < EPSILON || newHit.m == TABLE) return false;
				
				attenuation *= transmittance;
				pdf *= transmittance;
			}
			else
			{
				attenuation *= 1. - pPassThrough;
				pdf *= 1. - pPassThrough;
			}
		}
		case TABLE:
		case LABEL:
		{
			float skylightPdf = 0.;
			float bsdfPdf = 0.;
			vec3 bounceDirection;

			float c0;
			switch(MIS_TYPE)
			{
				case NO_MIS: c0 = 0.;
					break;
				case BALANCED_MIS: c0 = 0.5;
					break;
				//Rough estimation of relative importance between sunlight and skylight.
				case SMART_MIS:
				{
					const float skyWeight = 1.;
					float sunWeight = SUN_INTENSITY*max(0., dot(hit.n, -SUN_DIRECTION));
					float skyImportance = 2.*PI;
					float sunImportance = SUN_INTENSITY*sphereCapArea(SUN_APPARENT_RATIO*0.5*PI);
					c0 = sunImportance/(skyImportance + sunImportance);
				}
			}
			float c1 = 1. - c0;

			//Sunlight strategy.
			if(randUniform() < c0)
			{
				bounceDirection = randomSunlightDirection(randUniform(), randUniform(), skylightPdf, hit.n);
				bsdfPdf = max(0., dot(hit.n, bounceDirection)/PI);

				float p0 = c0*skylightPdf;
				float p1 = c1*bsdfPdf;
				float ps = p0 + p1;
				pdf *= p0;
				attenuation *= p0/ps;
			}
			//BSDF strategy.
			else
			{
				bounceDirection = randomLambertianReflection(hit.n, randUniform(), randUniform());
				bsdfPdf = dot(hit.n, bounceDirection)/PI;
				skylightPdf = randomSunlightPdf(bounceDirection);

				float p0 = c0*skylightPdf;
				float p1 = c1*bsdfPdf;
				float ps = p0 + p1;
				pdf *= p1;
				attenuation *= p1/ps;
			}

			float d = max(0., dot(hit.n, bounceDirection));
			//if(d <= 0.) return false;

			attenuation *= d / PI * color;
			ro = hit.p + hit.n*2.*EPSILON;
			rd = bounceDirection;
			return true;
		}
		//Glass.
		case FLASK:
		case CAP:
		{
			const float airRefractionIndex = 1.;
			const float glassRefractionIndex = 1.5;
			float refractionIndexRatio = airRefractionIndex/glassRefractionIndex;
			
			bool backFace = dot(rd, hit.n) > 0.;
			//bool backFace = hit.d < 0.;
			if(backFace) //swap for backface
			{
				refractionIndexRatio = 1./refractionIndexRatio;
				hit.n = -hit.n;
			}
			
			float cosTheta = dot(-rd, hit.n);
			float sinTheta = sqrt(1. - sqr(cosTheta));
			float reflectedRatio = sinTheta * refractionIndexRatio > 1. ? 1. : reflectance(cosTheta, refractionIndexRatio);
			
			//Total internal reflection, or both reflection and refraction but with Schlik's approximation giving the reflectance (that cancels out by stochastic selection).
			if(reflectedRatio == 1. || reflectedRatio > randUniform())
			{
				attenuation *= reflectedRatio;
				pdf *= reflectedRatio;
				rd = reflect(rd, hit.n);
			}
			else
			{
				attenuation *= (1. - reflectedRatio);
				pdf *= (1. - reflectedRatio);
				rd = refract(rd, hit.n, refractionIndexRatio);
			}
			
			ro = hit.p - hit.n*2.*EPSILON;
			attenuation *= color;
			return true;
		}
	}
	return false;
}

//----------------------------------------------------

mat3 observerViewMatrix(in vec2 mouseUV)
{
	vec2 shift = vec2(0., -0.5);
	if(mouseUV == vec2(0., 0.))
		shift += vec2(0.5);
	float a = (shift.x + mouseUV.x)*PI*2.;
	float b = (shift.y + mouseUV.y)*PI;
	
	vec3 camera = vec3(cos(b)*sin(a), sin(b), cos(b)*cos(a));
	//Z vector
	vec3 up = normalize(cross(cross(camera, vec3(0, 1, 0)), camera));
	//Y vector
	vec3 x = normalize(cross(up, camera));
	
	const float depth = 0.5;
	return depth*mat3(x, up, camera);
}
void pixelRay(in vec2 ij, out vec3 ro, out vec3 rd)
{
	//Towards -Z.
	ro = vec3(0, 0, 1);
	vec2 q = (ij - 0.5*iResolution.xy)/iResolution.y;
	rd = normalize(vec3(q, 0) - ro);

	mat3 view = observerViewMatrix(CONTROL_CAMERA ? iMouse.xy/iResolution.xy : vec2(0.));
	ro = view[2];
	rd = normalize(view*rd);
}


void render(out vec4 fragColor, in vec2 fragCoord)
{
	fragColor = vec4(vec3(0.), 1.);

	vec3 ro;
	vec3 rd;
	pixelRay(fragCoord.xy + vec2(randUniform(), randUniform()), ro, rd);
	vec3 attenuation = vec3(1.);
	vec3 pdf = vec3(1.);

	for(uint b = BOUNCES; b > ZEROu; b--)
	{
		Hit hit;
		if(!intersect(ro, rd, hit))
		{
			fragColor.xyz += attenuation/pdf * background(rd);
			break;
		}
		
		hit.n = calcNormal(hit.p - rd*2.*EPSILON, 0.0001);
		
		
		if(!scatter(ro, rd, hit, attenuation, pdf))
			break;
	}
}



//----------------------------------------------------

void mainImage(out vec4 fragColor, in vec2 fragCoordv)
{
	vec2 pixel = fragCoordv;

	vec4 oldColor = texelFetch(iChannel0, ivec2(fragCoordv), 0);
	if(iMouse.z > 0.) oldColor = vec4(0.);
	for(uint i = ZEROu; i < SAMPLES_PER_FRAME; i++)
	{
		initRandState(iTime, uvec2(pixel), uvec2(iResolution.xy), uint(oldColor.w));
		
		vec4 newColor;
		render(newColor, pixel);
		
		//Avoid degenerate paths.
		if(any(isnan(newColor)) || any(lessThan(newColor, vec4(0.))))
			continue;

		float count = oldColor.a + newColor.a;
		oldColor = vec4(oldColor.rgb + (newColor.rgb - oldColor.rgb) / count, count);
		fragColor = oldColor;
	}

	fragColor = oldColor;
}


