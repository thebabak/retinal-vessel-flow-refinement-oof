"""Fixed arm definitions; importing this file does not import PyTorch."""
from dataclasses import asdict, dataclass

SEEDS = [42, 123, 2026, 3000, 3001]
DATASETS = ["CHASE", "DRIVE", "HRF"]
EPOCHS = 120
STEPS = 8
ALPHAS = [0., .25, .5, .75, 1.]

@dataclass(frozen=True)
class Arm:
    name: str
    family: str = "grid"
    use_gate: bool = False
    use_fm: bool = False
    selective_target: bool = False
    selective_segmentation: bool = False
    dice_weight: float = .5
    thin_weight: float = 0.
    anchor_weight: float = 0.
    augmentation: bool = True
    use_rgb: bool = True
    patch_grid: str = "native"
    vessel_probability: float = .5
    degrade_rgb: bool = False
    degrade_gt: bool = False
    comparator: str = "direct"
    purpose: str = ""

    @property
    def direct(self): return not self.use_fm
    def record(self): return asdict(self)

ARMS = [
    Arm("o1_thin_only", thin_weight=.5,
        purpose="O1 recovery term alone; compare with original Direct and retained O1 both terms."),
    Arm("o1_anchor_only", anchor_weight=1.,
        purpose="O1 correct-pixel anchor alone; fixed weight matches retained O1."),
    Arm("o2_endpoint_only", use_gate=True, use_fm=True, selective_target=True,
        comparator="esfm", purpose="Full-support endpoint alone, original ESFM gate/target/FM retained; anchor zero."),
    Arm("o2_anchor_only", use_gate=True, use_fm=True, selective_target=True,
        selective_segmentation=True, anchor_weight=.1, comparator="esfm",
        purpose="Anchor alone on real endpoint; original selective-composite segmentation retained."),
    Arm("direct_dice_0", dice_weight=0., purpose="Only Direct Dice coefficient differs; old .5 retained."),
    Arm("direct_dice_025", dice_weight=.25, purpose="Only Direct Dice coefficient differs; old .5 retained."),
    Arm("direct_dice_1", dice_weight=1., purpose="Only Direct Dice coefficient differs; old .5 retained."),
    Arm("direct_no_rgb", use_rgb=False,
        purpose="Replace RGB by .5 before both velocity and auxiliary classifier; normalized RGB is zero; parameter budget unchanged."),
    Arm("direct_no_augmentation", augmentation=False,
        purpose="Disable aligned rotations/flips only; consume original RNG draws to retain paired image order."),
    Arm("o2_no_rgb", use_gate=True, use_fm=True, selective_target=True,
        anchor_weight=.1, use_rgb=False, comparator="esfm_endpoint_anchor",
        purpose="Only RGB conditioning removed from retained full-endpoint O2+anchor; normalized RGB zero for velocity and gate."),
    Arm("patch_512", family="patch", patch_grid="512", comparator="native_patch_direct",
        purpose="Same 256 tiles/128 stride/Hann/sampling rule/patch count/batch/loss on original 512 fields."),
    Arm("native_uniform", family="patch", vessel_probability=0., comparator="native_patch_direct",
        purpose="Uniform-FOV centres only; native image/GT/loss/fusion unchanged."),
    Arm("native_no_augmentation", family="patch", augmentation=False, comparator="native_patch_direct",
        purpose="Native O3 without aligned rotation/flip; consume all transform RNG draws."),
    Arm("native_degraded_rgb", family="patch", degrade_rgb=True, comparator="native_patch_direct",
        purpose="Native-grid patches; retained 512 RGB bilinearly projected to native dimensions; native GT fixed."),
    Arm("native_degraded_gt", family="patch", degrade_gt=True, comparator="native_patch_direct",
        purpose="Native-grid patches; retained 512 GT projected to native dimensions; native RGB fixed."),
    Arm("native_degraded_both", family="patch", degrade_rgb=True, degrade_gt=True,
        comparator="native_patch_direct", purpose="Both 512 RGB and GT projected to native grid; same original-native sampling pools."),
]
BY_NAME = {a.name: a for a in ARMS}
assert len(BY_NAME) == 16

def plan():
    return dict(datasets=DATASETS, seeds=SEEDS, arms=[a.record() for a in ARMS],
        new_final_fits=len(ARMS)*len(DATASETS)*len(SEEDS), epochs=EPOCHS, steps=STEPS,
        baseline_seed=42, new_baseline_fits=0, coarse_maps="retained audited OOF seed42 training maps; frozen full-baseline validation/test maps",
        optimizer="AdamW lr3e-4 weight_decay1e-4 cosine120 epochs FP32; separate branch gradient clipping1; no warmup",
        grid_sampling="full512 images batch4, shuffled per epoch; original paired RNG seed+91001/time seed+91002",
        patch_sampling="four256 patches per training image/epoch, batch16; native RNG seed+141001; fixed stride128 positiveHann inference batch16",
        degradation="RGB retained512->native bilinear align_corners=False; GT nearest-exact CHASE/DRIVE, nearest HRF; FOV/nativecoarse unchanged",
        information_factor_sampling="Original native GT/FOV defines every centre pool in degraded RGB/GT arms; no recomputation from degraded GT",
        settings_selection="All coefficients/settings fixed here; no new pilot/search. Learned checkpoint maximum raw validation mean image Dice at epochs10..120; strict improvement1e-12/earlier tie. Secondary blend picks alpha0,.25,.5,.75,1, including identity. All240choices locked before any NEW test evaluation.",
        inference="GT-free coarse+RGB only. Grid direct1velocitycall; FM8Euler+1gate. Patches fuse FP32 logit residuals with positiveHann then project residual only to original512 coarse.",
        outcomes="Same retained original512 GT/FOV, threshold0.5, original training-only thin cutoff; FP32 baseline/raw/blended probabilities and GT/FOV saved. Native raw residuals also saved; blend residual only if different selected checkpoint. Native probability arrays not duplicated; native coarse+residual+source retain inference provenance.",
        prespecified_contrasts=[
            "O1 2x2: retained Direct, thin-only, anchor-only, retained thin+anchor",
            "O2 2x2: retained ESFM, endpoint-only, anchor-only, retained endpoint+anchor",
            "Direct Dice weights0,.25,.5(retained),1; no validation choice of best test-performing weight",
            "Direct no-RGB minus retained Direct; O2 no-RGB minus retained O2; Direct no-augmentation minus retained Direct",
            "patch512/native-uniform/native-noaugmentation minus retained O3",
            "Native information2x2: retained O3 nativeRGB/nativeGT; degradedRGB; degradedGT; degradedboth"],
        inference_limits=["One baseline and one split per dataset; not five full pipeline fits", "Previously examined cohorts; new all-fit lock does not erase historical test exposure", "512patch/native changes physical field of view; information2x2 holds grid/context/sampling/fusion fixed but is not a clinical vessel-resolution experiment", "Canonical DRIVEtest/HRFconversion upstream provenance unresolved", "Old controls are reused and are not fresh repetitions of control training"],
        preserve_previous=True)
