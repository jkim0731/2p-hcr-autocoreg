import os, sys, json, time
os.environ.setdefault('MFISH_CACHE_DIR', '/scratch/tmp/mfish_cache'); os.makedirs('/scratch/tmp/mfish_cache', exist_ok=True)
D='/root/capsule/data/'
os.environ['MFISH_HCR_DIR']=D+'HCR_837568-01_2026-05-20_13-00-00_processed_2026-06-04_19-00-46'
os.environ['MFISH_CZ_SEG_TIF']=D+'multiplane-ophys_837568_2026-02-13_14-38-22_cortical-zstack-segmentation_2026-08-04_04-38-48/channel_0_ref_0/segmentation_masks.tif'
os.environ['MFISH_CZ_REG_TIF']=D+'multiplane-ophys_837568_2026-02-13_14-38-22_cortical-zstack-registration_2026-08-03_22-44-14/cortical_zstack_0/channel_0_ref_0/cortical_zstack_0_2xREG.tif'
os.environ['MFISH_PANNEURONAL_SIDS']='837568'
sys.path.insert(0,'/2p2fish/src')
t=time.time()
from autocoreg.io.subjects import load_subject
s=load_subject('837568'); print('loaded', time.time()-t, len(s.cz_centroids), len(s.hcr_centroids), flush=True)
from autocoreg.initial_registration.cz_surface import compute_cz_pia_surface_panneuronal
cz=compute_cz_pia_surface_panneuronal(s); print({k:v for k,v in cz.items()}, flush=True)
