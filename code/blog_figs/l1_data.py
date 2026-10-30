"""Collect the data for the L1/L2 figure (subject 839909), using the pipeline's own functions.
Saves /scratch/tmp/diagram/l1_839909.npz"""
import os, sys, json, time
os.environ.setdefault('MFISH_CACHE_DIR', '/scratch/tmp/mfish_cache'); os.makedirs('/scratch/tmp/mfish_cache', exist_ok=True)
D = '/root/capsule/data/'
HCR = D + 'HCR_839909_2026-07-23_13-00-00_processed_2026-07-30_21-17-42'
SEG = D + 'multiplane-ophys_839909_2026-05-07_700x700-Green-VISp_cortical-zstack-segmentation_2026-08-25_21-12-00/channel_0_ref_0/segmentation_masks.tif'
REG = D + 'multiplane-ophys_839909_2026-05-07_700x700-Green-VISp_cortical-zstack-registration_2026-08-25_07-30-31/cortical_zstack_0/channel_0_ref_0/cortical_zstack_0_2xREG.tif'
A = D + 'HCR_839909_2026-07-23_13-00-00_2p-3DFISH-autocoreg_2026-09-03_16-08-39/'
os.environ.update(MFISH_HCR_DIR=HCR, MFISH_CZ_SEG_TIF=SEG, MFISH_CZ_REG_TIF=REG, MFISH_PANNEURONAL_SIDS='839909')
sys.path.insert(0, '/2p2fish/src')
import numpy as np, tifffile, zarr
t0 = time.time()
from autocoreg.io.subjects import load_subject
from autocoreg.io.centroids import centroids_um
from autocoreg.io.hcr_image import depth_from_surface
from autocoreg.io.inputs import strict_gfp_ids
from autocoreg.initial_registration import cz_surface as CS
from autocoreg.initial_registration import surface_detect as SD
s = load_subject('839909'); print('loaded', time.time() - t0, len(s.cz_centroids), len(s.hcr_centroids), flush=True)
# ---- 2p: density profile after detrending the fitted tilt plane
cz_um, _ = centroids_um(s, 'cz'); z, y, x = cz_um[:, 0], cz_um[:, 1], cz_um[:, 2]
pia = json.load(open(A + 'registration/surfaces/839909_cz_onset_panneuronal.json'))
a, b = pia['a'], pia['b']
depth2, cent2, dens2 = CS._panneuronal_density_profile(z, y, x, a, b, CS.PANNEURONAL_DENS_BIN_UM, CS.PANNEURONAL_DENS_SIGMA_BINS)
ton, l12, sharp, steep = CS._panneuronal_l1l2_onset(z, y, x, a, b)
print('2p tissue onset', ton, 'L1/L2 onset', l12, 'json', pia['tissue_onset_um'], pia['l1l2_onset_um'], flush=True)
# ---- HCR: GFP+ density below the HCR pia
hs = json.load(open(A + 'registration/surfaces/839909_hcr_top_iter07.json'))
hzyx, hids = centroids_um(s, 'hcr_all')
gfp, meta = strict_gfp_ids('839909', force_gmm=True)
keep = np.isin(hids, np.fromiter(gfp, dtype=np.int64, count=len(gfp)))
hd = depth_from_surface(hzyx[keep][:, [2, 1, 0]], hs)
on = SD.hcr_l1l2_onset_from_depths(hd); print('HCR onset', on, 'n gfp', keep.sum(), 'of', len(hids), flush=True)
edges = np.arange(SD.HCR_L1L2_DEPTH_LO_UM, SD.HCR_L1L2_DEPTH_HI_UM + SD.HCR_L1L2_DEPTH_BIN_UM, SD.HCR_L1L2_DEPTH_BIN_UM)
from scipy.ndimage import gaussian_filter1d
cent_h = 0.5 * (edges[:-1] + edges[1:]); dens_h = gaussian_filter1d(np.histogram(hd[np.isfinite(hd)], bins=edges)[0].astype(float), SD.HCR_L1L2_SMOOTH_SIGMA_BINS)
# ---- side-view bands
vol = tifffile.imread(REG, key=None); print('2p stack', vol.shape, flush=True)
ym = vol.shape[1] // 2; band2p = vol[:, ym - 25:ym + 25, :].max(1)
zz = zarr.open(HCR + '/image_tile_fusing/fused/channel_405.zarr', mode='r')['4'][0, 0]
yh = zz.shape[1] // 2; bandh = np.asarray(zz[:, yh - 15:yh + 15, :]).max(1)
xh = np.arange(bandh.shape[1]) * 3.9208892743996073
zs_h = -depth_from_surface(np.column_stack([xh, np.full_like(xh, yh * 3.9208892743996073), np.zeros_like(xh)]), hs)   # surface z at the band centre
np.savez('/scratch/tmp/diagram/l1_839909.npz', cent2=cent2, dens2=dens2, ton=ton, l12=l12, c=pia['c'], a=a, b=b, ym_um=ym * 0.78,
         cent_h=cent_h, dens_h=dens_h, hcr_onset=on['onset_um'], hcr_steep=on['steepest_rise_um'], band2p=band2p, bandh=bandh,
         zs_h=zs_h, xh=xh, n_gfp=int(keep.sum()), n_all=len(hids))
print('done', time.time() - t0)
