"""White-tray colour constancy + foreground statistics + Lab prototype matching.

MicroPython runtime: math/lists only, no OpenCV/scikit-image dependencies.
Sources and standard test-vector attribution: see THIRD_PARTY_NOTICES.md.
"""
import math
COLOR_ALGORITHM_VERSION='white-lab-foreground-20261014-v1'
COLOR_CODES=('RED','ORANGE','YELLOW','GREEN','CYAN','BLUE','PURPLE','BLACK','WHITE','GRAY','UNKNOWN')
COLOR_NAMES_ZH=('红色','橙色','黄色','绿色','青色','蓝色','紫色','黑色','白色','灰色','未知色')
PROFILE={'version': 'white-lab-20261014-v1', 'metric': 'hybrid', 'raw_weight': 0.0, 'max_distance': 15.0, 'min_gap': 1.0, 'camera_config_changed': False, 'train_objects': 425, 'validation_objects': 178, 'fitted_on': '425 train colour crops, including 9 current labelled crops; no shape class used during inference'}
PROTOTYPES={'BLUE': [{'lab': [56.47160883291487, -21.94013641404935, -13.25114312933362], 'lab_normalized': [64.05052337497595, -24.518716056493194, -14.402505514044805], 'group': 'legacy_old_initial_0', 'training_examples': 3}, {'lab': [48.78942933917758, -1.0263656048656034, -34.6826543201299], 'lab_normalized': [50.394388916403265, -1.0230051373861793, -35.44947975535058], 'group': 'legacy_new_initial_0', 'training_examples': 4}, {'lab': [29.071851915441222, 19.31972979989721, -44.77882140035351], 'lab_normalized': [38.55875202810139, 23.86196553511491, -54.33571754456542], 'group': 'current_initial_0', 'training_examples': 3}, {'lab': [47.876580607859154, 2.7730615553431814, -39.12833153579167], 'lab_normalized': [47.874235233055074, 2.769328459080511, -39.43111539223454], 'group': 'current_initial_1', 'training_examples': 3}, {'lab': [46.24921880260871, 3.778883545075634, -39.480594533517355], 'lab_normalized': [46.84062697281721, 3.8405017449522902, -41.078794122340135], 'group': 'supplement_6', 'training_examples': 40}, {'lab': [46.75349307490933, 3.5729222895348745, -39.69298583341556], 'lab_normalized': [47.56565896293186, 3.6567422162188112, -39.92318031608557], 'group': 'supplement_5', 'training_examples': 39}], 'PURPLE': [{'lab': [38.74086310011755, 15.82144735402316, -36.82704015950639], 'lab_normalized': [45.893518766513736, 18.35717760914557, -42.45438637725074], 'group': 'legacy_old_initial_1', 'training_examples': 3}, {'lab': [38.23434007187167, 19.65912842682059, -39.65264832507383], 'lab_normalized': [44.35161005599595, 22.368998401590375, -44.90545639972602], 'group': 'legacy_new_initial_1', 'training_examples': 4}, {'lab': [34.03138906989362, 20.121103489938562, -39.31558953994929], 'lab_normalized': [42.55723816167596, 24.72267028946973, -47.793650890518435], 'group': 'current_initial_2', 'training_examples': 3}, {'lab': [38.74237088443398, 21.510349590723095, -39.26605145602436], 'lab_normalized': [44.59058123379088, 24.264456896452792, -44.49570749547833], 'group': 'prior_pose_2', 'training_examples': 36}, {'lab': [36.78020721014082, 21.642009237185967, -41.36196003870836], 'lab_normalized': [43.61372669086752, 24.296507386725175, -46.09138206292366], 'group': 'prior_pose_5', 'training_examples': 36}, {'lab': [33.26677605435965, 21.123216958232245, -39.42556021318596], 'lab_normalized': [42.35040641450309, 25.979181310432644, -48.11888470987955], 'group': 'historical_purple', 'training_examples': 1}], 'RED': [{'lab': [36.40331934020102, 45.15640912205981, 17.134638804677248], 'lab_normalized': [43.02041731713708, 50.20895435787195, 19.672543451381188], 'group': 'supplement_3', 'training_examples': 38}, {'lab': [29.69197461067217, 34.09346602307075, 10.05644967620834], 'lab_normalized': [44.49617361021569, 47.679644174052044, 14.51114435862193], 'group': 'supplement_4', 'training_examples': 40}], 'GREEN': [{'lab': [39.17395260688986, -20.08089831342874, 6.429292030322981], 'lab_normalized': [67.59726910076361, -32.687341881766336, 10.816711212424456], 'group': 'supplement_1', 'training_examples': 40}, {'lab': [31.104627574214227, -17.460849032540608, 5.2620589097674975], 'lab_normalized': [67.40434438719348, -33.89815372578225, 10.094403805870055], 'group': 'supplement_0', 'training_examples': 59}], 'YELLOW': [{'lab': [76.82779122122612, -14.369488737530578, 61.95359411649487], 'lab_normalized': [70.65703960998052, -13.097834793581399, 57.50494138945764], 'group': 'supplement_2', 'training_examples': 37}, {'lab': [77.7813333735858, -14.019969855028375, 62.77656883446842], 'lab_normalized': [70.46554207299695, -12.956290876222827, 57.30171811935476], 'group': 'prior_pose_3', 'training_examples': 36}]}
PALETTE={'RED':(210,30,30),'ORANGE':(230,125,25),'YELLOW':(220,210,25),'GREEN':(35,160,65),'CYAN':(25,180,180),'BLUE':(35,85,210),'PURPLE':(145,60,185)}
def _median(values):
    values=sorted(values);n=len(values)
    return values[n//2] if n%2 else (values[n//2-1]+values[n//2])*.5
def rgb_to_lab(rgb):
    values=[]
    for channel in rgb:
        x=max(0.,min(1.,channel/255.));values.append(x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4)
    r,g,b=values
    xyz=((.4124564*r+.3575761*g+.1804375*b)/.95047,.2126729*r+.7151522*g+.0721750*b,(.0193339*r+.1191920*g+.9503041*b)/1.08883)
    f=[x**(1./3.) if x>.008856451679 else 7.787037037*x+16./116. for x in xyz]
    return (116*f[1]-16,500*(f[0]-f[1]),200*(f[1]-f[2]))
def delta_e00(first,second,k_l=2.0):
    l1,a1,b1=first;l2,a2,b2=second
    c1=math.sqrt(a1*a1+b1*b1);c2=math.sqrt(a2*a2+b2*b2);c=(c1+c2)*.5;c7=c**7
    g=.5*(1-math.sqrt(c7/(c7+6103515625.)))
    p1=(1+g)*a1;p2=(1+g)*a2;cp1=math.sqrt(p1*p1+b1*b1);cp2=math.sqrt(p2*p2+b2*b2)
    h1=math.atan2(b1,p1)*180/math.pi%360;h2=math.atan2(b2,p2)*180/math.pi%360
    dh=h2-h1
    if cp1*cp2==0:dh=0.
    elif dh>180:dh-=360
    elif dh< -180:dh+=360
    dl=l2-l1;dc=cp2-cp1;dH=2*math.sqrt(cp1*cp2)*math.sin(dh*math.pi/360)
    ml=(l1+l2)*.5;mc=(cp1+cp2)*.5
    if cp1*cp2==0:mh=h1+h2
    elif abs(h1-h2)<=180:mh=(h1+h2)*.5
    elif h1+h2<360:mh=(h1+h2+360)*.5
    else:mh=(h1+h2-360)*.5
    rad=math.pi/180
    t=1-.17*math.cos((mh-30)*rad)+.24*math.cos(2*mh*rad)+.32*math.cos((3*mh+6)*rad)-.20*math.cos((4*mh-63)*rad)
    sl=1+.015*(ml-50)**2/math.sqrt(20+(ml-50)**2);sc=1+.045*mc;sh=1+.015*mc*t
    rc=2*math.sqrt(mc**7/(mc**7+6103515625.));rt=-rc*math.sin(60*math.exp(-((mh-275)/25)**2)*rad)
    x=dl/(k_l*sl);y=dc/sc;z=dH/sh
    return math.sqrt(max(0.,x*x+y*y+z*z+rt*y*z))
def _neutral_signature(points,minimum=12):
    eligible=[]
    for r,g,b in points:
        hi=max(r,g,b);lo=min(r,g,b)
        if 75<=hi<250 and hi-lo<=hi*.28 and g>0:eligible.append((hi,r/g,b/g))
    if len(eligible)<minimum:return None
    cutoff=sorted(p[0] for p in eligible)[int((len(eligible)-1)*.35)]
    subset=[p for p in eligible if p[0]>=cutoff]
    rg=_median([p[1] for p in subset]);bg=_median([p[2] for p in subset])
    if not .65<rg<1.5 or not .65<bg<1.5:return None
    return {'neutral_r_g':rg,'neutral_b_g':bg,'white_level':_median([p[0] for p in subset]),'gains':(1./rg,1.,1./bg),'neutral_points':len(subset)}
def color_context(pixels):
    h,w,c=pixels.shape
    if c!=3:raise ValueError('Expected RGB888')
    points=[]
    for row in range(20):
        y=row*(h-1)//19
        for col in range(24):
            x=col*(w-1)//23;points.append(tuple(int(pixels[y,x,k]) for k in range(3)))
    signature=_neutral_signature(points,24)
    if signature is None:return {'calibrated':False,'gains':(1.,1.,1.),'reason':'no_neutral_reference','white_level':200.}
    signature.update(calibrated=True,reason='neutral_reference');return signature
def _local_context(pixels,t,global_context):
    h,w,c=pixels.shape;x0=max(0,int(t['x']));y0=max(0,int(t['y']));x1=min(w,int(t['x']+t['w']));y1=min(h,int(t['y']+t['h']))
    dx=max(8,(x1-x0)//3);dy=max(8,(y1-y0)//3);left=max(0,x0-dx);right=min(w,x1+dx);top=max(0,y0-dy);bottom=min(h,y1+dy);points=[]
    for row in range(10):
        y=top+(2*row+1)*(bottom-top)//20
        for col in range(10):
            x=left+(2*col+1)*(right-left)//20
            if x0<=x<x1 and y0<=y<y1:continue
            points.append(tuple(int(pixels[y,x,k]) for k in range(3)))
    local=_neutral_signature(points,12)
    if local is None:return global_context
    if global_context.get('calibrated'):
        if abs(local['neutral_r_g']-global_context['neutral_r_g'])>.18 or abs(local['neutral_b_g']-global_context['neutral_b_g'])>.18:return global_context
        local['gains']=tuple(.75*a+.25*b for a,b in zip(local['gains'],global_context['gains']))
    local.update(calibrated=True,reason='local_neutral_reference');return local
def sample_object(pixels,target,context=None):
    h,w,c=pixels.shape
    if c!=3:raise ValueError('Expected RGB888')
    global_context=context if context is not None else color_context(pixels)
    local=_local_context(pixels,target,global_context);gains=local['gains']
    x0=max(0,int(target['x']));y0=max(0,int(target['y']));x1=min(w,int(target['x']+target['w']));y1=min(h,int(target['y']+target['h']))
    if x1-x0<3 or y1-y0<3:return None,{'reason':'small_box','calibrated':False}
    dx=(x1-x0)//25;dy=(y1-y0)//25;x0+=dx;x1-=dx;y0+=dy;y1-=dy
    side=20;points=[];mask=[]
    for row in range(side):
        y=y0+(2*row+1)*(y1-y0)//(2*side)
        for col in range(side):
            x=x0+(2*col+1)*(x1-x0)//(2*side)
            rgb=tuple(int(pixels[y,x,k]) for k in range(3));hi=max(rgb);lo=min(rgb)
            points.append(rgb);mask.append(hi>=25 and hi-lo>=hi*.18)
    seen=[False]*len(mask);best=[]
    for index in range(len(mask)):
        if not mask[index] or seen[index]:continue
        queue=[index];seen[index]=True;pos=0
        while pos<len(queue):
            j=queue[pos];pos+=1;r=j//side;col=j%side
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    if not dr and not dc:continue
                    nr=r+dr;nc=col+dc
                    if 0<=nr<side and 0<=nc<side:
                        k=nr*side+nc
                        if mask[k] and not seen[k]:seen[k]=True;queue.append(k)
        if len(queue)>len(best):best=queue
    diag={'calibrated':bool(local.get('calibrated')),'foreground_points':len(best),'total_points':len(mask),'white_gains':gains,'white_level':local['white_level'],'white_reference':local.get('reason')}
    if len(best)<max(6,int(len(mask)*.025)):
        levels=sorted(max(p) for p in points);chroma=sum(mask)
        if chroma>len(mask)*.1:return None,dict(diag,reason='fragmented_foreground')
        rgb=tuple(_median([p[k] for p in points]) for k in range(3));hi=max(rgb)
        neutral=7 if hi<45 else 8 if hi>=180 else 9
        return {'neutral_index':neutral,'rgb':rgb},dict(diag,reason='neutral_object')
    selected=[points[i] for i in best];levels=sorted(max(p) for p in selected);low=levels[int((len(levels)-1)*.10)];high=levels[int((len(levels)-1)*.90)]
    selected=[p for p in selected if low<=max(p)<=high]
    corrected=[tuple(max(0.,min(255.,p[k]*gains[k])) for k in range(3)) for p in selected]
    # Normalise each foreground point before combining faces. Independent
    # channel medians can synthesize a hue from different shadow/highlight
    # faces, which was especially unstable on dark blue plastic.
    ratios=[tuple(v/max(1.,max(p)) for v in p) for p in corrected]
    chroma=tuple(_median([p[k] for p in ratios]) for k in range(3))
    chroma=tuple(v/max(1e-6,max(chroma)) for v in chroma)
    peak=_median([max(p) for p in corrected]);rgb=tuple(v*peak for v in chroma)
    canonical=tuple(v*180 for v in chroma)
    relative=tuple(v*200/max(60.,local['white_level']) for v in rgb)
    result={'rgb':rgb,'lab':rgb_to_lab(relative),'lab_normalized':rgb_to_lab(canonical),'relative_value':peak/max(60.,local['white_level'])}
    return result,dict(diag,reason='foreground_statistics')
def prototype_distance(feature,reference):
    kind=PROFILE.get('metric','hybrid')
    if kind=='lab_raw':return delta_e00(feature['lab'],reference['lab'])
    if kind=='lab_normalized':return delta_e00(feature['lab_normalized'],reference['lab_normalized'])
    weight=PROFILE.get('raw_weight',.5)
    return weight*delta_e00(feature['lab'],reference['lab'])+(1-weight)*delta_e00(feature['lab_normalized'],reference['lab_normalized'])
def classify_feature(feature):
    if feature is None:return 10,0.,{'reason':'no_object_colour'}
    if 'neutral_index' in feature:return feature['neutral_index'],1.,{'reason':'neutral_object'}
    scores=[]
    for code in COLOR_CODES[:7]:
        refs=PROTOTYPES.get(code)
        if not refs:
            rgb=PALETTE[code];refs=[{'lab':rgb_to_lab(rgb),'lab_normalized':rgb_to_lab(tuple(v*180/max(rgb) for v in rgb))}]
        scores.append(min(prototype_distance(feature,ref) for ref in refs))
    order=sorted(range(7),key=lambda i:scores[i]);best,second=order[:2];distance=scores[best];gap=scores[second]-distance
    if distance>PROFILE.get('max_distance',30.):return 10,0.,{'reason':'colour_out_of_reference','distance':distance,'gap':gap}
    if gap<PROFILE.get('min_gap',1.):return 10,0.,{'reason':'ambiguous_colour','distance':distance,'gap':gap}
    strength=max(0.,min(1.,gap/(gap+distance+1e-6)))
    return best,strength,{'reason':'lab_reference_match','distance':distance,'gap':gap}
def color_index(r,g,b,context=None):
    gains=context.get('gains',(1.,1.,1.)) if context else (1.,1.,1.)
    rgb=tuple(v*g for v,g in zip((r,g,b),gains));hi=max(rgb);lo=min(rgb)
    if hi<25:return 7
    if hi-lo<hi*.18:return 8 if hi>=180 else 9
    feature={'lab':rgb_to_lab(rgb),'lab_normalized':rgb_to_lab(tuple(v*180/max(1.,hi) for v in rgb))}
    return classify_feature(feature)[0]
def attach_color(pixels,target,context=None):
    feature,diag=sample_object(pixels,target,context);index,support,decision=classify_feature(feature)
    target.update(color_hint=COLOR_CODES[index],color_name_zh=COLOR_NAMES_ZH[index],color_support=support,
        color_profile_applied=bool(diag.get('calibrated')),color_algorithm_version=COLOR_ALGORITHM_VERSION,
        color_reason=decision['reason'],color_distance=decision.get('distance',0.),color_margin=decision.get('gap',0.),
        color_foreground_points=diag.get('foreground_points',0),color_white_reference=diag.get('white_reference','none'))
    return target

_color_tracks=[]
_color_last_ms=None

def reset_color_history():
    global _color_tracks,_color_last_ms
    _color_tracks=[]
    _color_last_ms=None

def _box_overlap(a,b):
    x0,y0=max(a['x'],b['x']),max(a['y'],b['y'])
    x1,y1=min(a['x']+a['w'],b['x']+b['w']),min(a['y']+a['h'],b['y']+b['h'])
    intersection=max(0,x1-x0)*max(0,y1-y0)
    return intersection/max(1,a['w']*a['h']+b['w']*b['h']-intersection)

def stabilize_colors(targets):
    """Three-frame history; a color change requires two consecutive readings.

    Match by overlap/class for continuity only, never infer color from shape or
    position. Disappeared tracks are removed immediately; UNKNOWN resets votes.
    """
    global _color_tracks,_color_last_ms
    import time
    now=time.ticks_ms() if hasattr(time,'ticks_ms') else int(time.monotonic()*1000)
    if _color_last_ms is not None:
        elapsed=time.ticks_diff(now,_color_last_ms) if hasattr(time,'ticks_diff') else now-_color_last_ms
        if elapsed>1200 or elapsed<0:_color_tracks=[]
    _color_last_ms=now
    used=[];next_tracks=[]
    for target in targets:
        raw=target['color_hint'];support=target['color_support']
        target['color_hint_raw']=raw
        candidates=[]
        for index,track in enumerate(_color_tracks):
            if index not in used and target.get('class_id',-1)==track['box'].get('class_id',-1):
                overlap=_box_overlap(target,track['box'])
                if overlap>=.3:candidates.append((overlap,index))
        history=[];confirmed=None
        if candidates:
            index=max(candidates)[1];used.append(index)
            history=_color_tracks[index]['history'][:]
            confirmed=_color_tracks[index]['confirmed']
        if raw=='UNKNOWN':history=[];confirmed=None
        history=(history+[raw])[-3:]
        if confirmed is None and raw!='UNKNOWN':confirmed=raw
        if raw!='UNKNOWN' and len(history)>=2 and history[-2]==raw:confirmed=raw
        stable=raw if raw==confirmed else 'UNKNOWN'
        agreed=stable!='UNKNOWN'
        target['color_hint']=stable
        target['color_name_zh']=COLOR_NAMES_ZH[COLOR_CODES.index(stable)]
        target['color_support']=support if stable==raw else 0.0
        target['color_temporal_consistent']=agreed
        next_tracks.append({'box':{'x':target['x'],'y':target['y'],'w':target['w'],'h':target['h'],'class_id':target.get('class_id',-1)},'history':history,'confirmed':confirmed})
    _color_tracks=next_tracks
    return targets
