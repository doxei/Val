"""Chargement et validation de la configuration (tous les nombres viennent du YAML).

Toute clé inconnue est refusée (extra="forbid") pour qu'une faute de frappe dans le YAML
ne passe jamais silencieusement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------- cœur
class VariableSpec(_Strict):
    base: float = Field(ge=0.0, le=1.0)
    tau: float = Field(default=3600.0, gt=0.0)
    homeostatic: bool = True
    rise: float = Field(default=1.0, gt=0.0)
    noise: float = Field(default=0.0, ge=0.0)
    desc: str = ""


class CouplingSpec(_Strict):
    source: str
    target: str
    weight: float


class DriveSpec(_Strict):
    rest: float = Field(gt=0.0, lt=1.0)
    tau: float = Field(gt=0.0)
    desc: str = ""
    weights: dict[str, float] = {}


class NeedSpec(_Strict):
    grace_seconds: float = Field(ge=0.0)
    rise_tau: float = Field(gt=0.0)
    satiation_on_interaction: float = Field(default=0.0, ge=0.0, le=1.0)
    desc: str = ""


class CircadianSpec(_Strict):
    awake_start: str = "07:00"
    sleep_start: str = "23:30"
    energy_drain_per_hour: float = Field(default=0.03, ge=0.0)
    energy_recharge_tau: float = Field(default=10800.0, gt=0.0)
    energy_ceiling: float = Field(default=0.95, gt=0.0, le=1.0)
    energy_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    sleep_arousal_offset: float = 0.0
    night_wake_seconds: float = Field(default=1800.0, ge=0.0)


class HabituationSpec(_Strict):
    window_seconds: float = Field(default=600.0, gt=0.0)
    decay_per_repeat: float = Field(default=0.35, ge=0.0)


class ReboundSpec(_Strict):
    delay_seconds: float = Field(default=120.0, ge=0.0)
    fraction: float = Field(default=0.25, ge=0.0, le=1.0)


class PADSpec(_Strict):
    gain: float = Field(default=1.6, gt=0.0)
    valence: dict[str, float] = {}
    arousal: dict[str, float] = {}
    dominance: dict[str, float] = {}


class EmotionPrototype(_Strict):
    P: float
    A: float
    D: float
    cue: str | None = None
    cue_min: float | None = None


class EmotionSpec(_Strict):
    sleep_label: str = "sommeil"
    calm_radius: float = Field(default=0.15, ge=0.0)
    dominant_drive_min: float = 0.12
    cue_min_default: float = 0.12
    cue_bonus: float = 0.08
    prototypes: dict[str, EmotionPrototype]


class IntensityWord(_Strict):
    threshold: float
    word: str

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return {"threshold": float(value[0]), "word": str(value[1])}
        return value


class MoodSpec(_Strict):
    pull_tau: float = Field(default=3600.0, gt=0.0)
    return_tau: float = Field(default=14400.0, gt=0.0)
    default_scale: float = Field(default=0.3, ge=0.0)
    neutral_radius: float = Field(default=0.06, ge=0.0)
    intensity_words: list[IntensityWord] = []
    octants: dict[str, str]

    @model_validator(mode="after")
    def _check_octants(self) -> MoodSpec:
        expected = {f"{p}P{a}A{d}D" for p in "+-" for a in "+-" for d in "+-"}
        missing = expected - set(self.octants)
        if missing:
            raise ValueError(f"octants manquants : {sorted(missing)}")
        return self


class OrganBand(_Strict):
    lo: float
    hi: float
    text: str

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return {"lo": float(value[0]), "hi": float(value[1]), "text": str(value[2])}
        return value


class OrganSpec(_Strict):
    desc: str = ""
    bias: float = 0.0
    sources: dict[str, float] = {}
    low: float | None = None
    high: float | None = None
    texts: list[OrganBand]
    # Boucle fermée (avenant 4 §1) : l'organe a son inertie (il « garde le poids ») et
    # renvoie son état vers les variables du cœur.
    rise_tau: float = Field(default=1.0, gt=0.0)    # secondes pour se mettre en place
    fall_tau: float = Field(default=1.0, gt=0.0)    # secondes pour se dénouer
    feedback: dict[str, float] = {}                 # variable → poids (sur l'écart au repos)


class StimulusSpec(_Strict):
    impulses: dict[str, float]
    rebound: dict[str, float] | None = None

    @model_validator(mode="before")
    @classmethod
    def _short_form(cls, value: Any) -> Any:
        if isinstance(value, dict) and value and "impulses" not in value:
            return {"impulses": value}
        return value


class HeartConfig(_Strict):
    tick_seconds: float = Field(default=1.0, gt=0.0)
    rng_seed: int = 1337
    max_time_step: float = Field(default=60.0, gt=0.0)
    catch_up_max_seconds: float = Field(default=259200.0, ge=0.0)
    save_every_seconds: float = Field(default=60.0, gt=0.0)
    journal_state_every_seconds: float = Field(default=120.0, gt=0.0)
    impulse_cap: float = Field(default=0.6, gt=0.0)
    sensitivity_floor: float = Field(default=0.2, ge=0.0, le=1.0)
    variables: dict[str, VariableSpec]
    couplings: list[CouplingSpec] = []
    drive_gain: float = Field(default=4.0, gt=0.0)
    drives: dict[str, DriveSpec]
    needs: dict[str, NeedSpec] = {}
    rest_reference: float = Field(default=0.75, gt=0.0, le=1.0)
    circadian: CircadianSpec = CircadianSpec()
    habituation: HabituationSpec = HabituationSpec()
    rebound: ReboundSpec = ReboundSpec()
    pad: PADSpec = PADSpec()
    emotions: EmotionSpec
    mood: MoodSpec
    organs: dict[str, OrganSpec] = {}
    stimuli: dict[str, StimulusSpec] = {}

    @model_validator(mode="after")
    def _check_references(self) -> HeartConfig:
        if "energy" not in self.variables:
            raise ValueError("la variable 'energy' est obligatoire (rythme veille/sommeil)")
        need_names = {f"need_{n}" for n in self.needs}
        targets = set(self.variables) | set(self.drives) | need_names
        for c in self.couplings:
            if c.source not in self.variables or c.target not in self.variables:
                raise ValueError(f"couplage inconnu : {c.source} -> {c.target}")
        for name, d in self.drives.items():
            for src in d.weights:
                if src not in self.variables and src not in need_names:
                    raise ValueError(f"drive {name} : source inconnue '{src}'")
        for name, s in self.stimuli.items():
            for tgt in list(s.impulses) + list(s.rebound or {}):
                if tgt not in targets:
                    raise ValueError(f"stimulus {name} : cible inconnue '{tgt}'")
        for name, o in self.organs.items():
            for tgt, w in o.feedback.items():
                spec = self.variables.get(tgt)
                if spec is None or not spec.homeostatic:
                    raise ValueError(f"organe {name} : retour vers '{tgt}' impossible "
                                     "(variable homéostatique attendue)")
                if abs(w) > 0.2:
                    raise ValueError(f"organe {name} : retour {tgt}={w} trop fort (|w| ≤ 0,2, "
                                     "la boucle doit rester stable)")
        cues = set()
        for v in self.variables:
            cues |= {v, f"{v}_dev", f"{v}_low"}
        for d in self.drives:
            cues |= {d, f"{d}_delta", f"{d}_exc"}
        cues |= need_names | {"need_rest"}
        for name, p in self.emotions.prototypes.items():
            if p.cue is not None and p.cue not in cues:
                raise ValueError(f"émotion {name} : indice inconnu '{p.cue}'")
        return self


# ----------------------------------------------------------------------- tempérament
class TemperamentProfile(_Strict):
    desc: str = ""
    big_five: dict[str, float] = {}
    sensitivity: float = Field(default=1.0, gt=0.0)
    bases: dict[str, float] = {}
    drive_rest: dict[str, float] = {}


class TemperamentSpec(_Strict):
    default_profile: str
    mehrabian: dict[str, dict[str, float]]
    profiles: dict[str, TemperamentProfile]

    @model_validator(mode="after")
    def _check(self) -> TemperamentSpec:
        if self.default_profile not in self.profiles:
            raise ValueError(f"profil par défaut inconnu : {self.default_profile}")
        if set(self.mehrabian) != {"P", "A", "D"}:
            raise ValueError("mehrabian doit définir exactement P, A et D")
        return self


# ------------------------------------------------------------------------ initiative
class InitiativeConfig(_Strict):
    urges: dict[str, dict[str, float]]
    threshold: float = Field(default=0.55, gt=0.0)
    rearm_below: float = Field(default=0.40, ge=0.0)
    min_interval_seconds: float = Field(default=2700.0, ge=0.0)
    quiet_start: str = "22:30"
    quiet_end: str = "09:00"
    max_unanswered: int = Field(default=2, ge=0)
    ignored_after_seconds: float = Field(default=900.0, ge=0.0)
    ignored_stimulus: str | None = "ignored"


class ThoughtsConfig(_Strict):
    """Pensée de fond (avenant 4 §6)."""
    enabled: bool = True
    db: str = "pensees.db"
    idle_seconds: float = Field(default=180.0, ge=0.0)      # silence avant de penser
    min_interval: float = Field(default=600.0, ge=0.0)
    per_hour: int = Field(default=4, ge=0)
    max_tokens: int = Field(default=400, ge=50)
    energy_cost: float = Field(default=0.003, ge=0.0)
    reappraisal_gain: float = Field(default=0.6, ge=0.0, le=2.0)
    calm: dict[str, float] = {}
    stir: dict[str, float] = {}
    rumination_window: float = Field(default=3600.0, ge=0.0)
    rumination_max: int = Field(default=2, ge=0)
    idea_similarity: float = Field(default=0.8, ge=0.0, le=1.0)
    propose_min_score: float = Field(default=0.6, ge=0.0, le=1.0)
    block_ideas: int = Field(default=2, ge=0, le=5)
    curiosity: bool = True                                   # creuse ses connaissances


# ------------------------------------------------------------------------- divers
class MemoryConfig(_Strict):
    recall_weights: dict[str, float] = {}
    recency_half_life_hours: float = 24.0


class ReinstateSpec(_Strict):
    threshold: float = Field(default=0.25, ge=0.0, le=1.0)   # |P| du souvenir pour réagir
    gain: float = Field(default=0.5, ge=0.0, le=2.0)
    positive: dict[str, float] = {}
    negative: dict[str, float] = {}


class EpisodicConfig(_Strict):
    """Mémoire épisodique et contexte temporel (avenant 3 §2, avenant 4 §7)."""
    db: str = "episodes.db"
    dim: int = Field(default=512, ge=64, le=4096)
    scales_seconds: list[float] = [30.0, 300.0, 3600.0, 86400.0]
    min_step_seconds: float = Field(default=10.0, gt=0.0)
    episode_gap_seconds: float = Field(default=1800.0, gt=0.0)
    w_semantic: float = 1.0
    w_context: list[float] = [0.25, 0.25, 0.15, 0.1]
    w_base: float = 0.05
    w_activation: float = 0.4
    w_mood: float = 0.15
    activation_tau_seconds: float = Field(default=120.0, gt=0.0)
    spread_forward: float = Field(default=0.5, ge=0.0, le=1.0)
    spread_backward: float = Field(default=0.3, ge=0.0, le=1.0)
    spread_reach: int = Field(default=2, ge=0, le=10)
    working_memory: int = Field(default=4, ge=0, le=10)
    working_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    recall_k: int = Field(default=3, ge=0, le=12)
    min_semantic: float = Field(default=0.22, ge=0.0, le=1.0)
    exclude_recent_seconds: float = Field(default=600.0, ge=0.0)
    resume_hours: float = Field(default=72.0, ge=0.0)
    preload_hours: float = Field(default=6.0, ge=0.0)
    preload_turns: int = Field(default=6, ge=0, le=40)
    reinstate: ReinstateSpec = ReinstateSpec()
    appraisal_pad: dict[str, list[float]] = {}   # stimulus → [P, A, D] (souvenirs importés)

    @model_validator(mode="after")
    def _check(self) -> EpisodicConfig:
        if len(self.w_context) != len(self.scales_seconds):
            raise ValueError("episodic : w_context doit avoir une valeur par échelle")
        for k, v in self.appraisal_pad.items():
            if len(v) != 3:
                raise ValueError(f"episodic.appraisal_pad.{k} : [P, A, D] attendu")
        return self


class StorageConfig(_Strict):
    dir: str = "data"
    db: str = "valdar.db"
    journal: str = "journal.jsonl"
    journal_max_bytes: int = Field(default=20_000_000, gt=0)
    journal_keep: int = Field(default=5, ge=0)


class SimConfig(_Strict):
    min_interval: float = 120.0
    max_interval: float = 2400.0
    absence_probability: float = Field(default=0.08, ge=0.0, le=1.0)
    absence_min: float = 7200.0
    absence_max: float = 21600.0
    night_events: bool = False
    event_types: list[str]
    weights: list[float]
    social_events: list[str] = []
    sampling_interval: float = 60.0
    bound_margin: float = 0.02
    stuck_limit: float = 0.05
    min_distinct_emotions: int = 6
    catch_up_tolerance: float = 0.03

    @model_validator(mode="after")
    def _check(self) -> SimConfig:
        if len(self.event_types) != len(self.weights) or not self.event_types:
            raise ValueError("sim : event_types et weights doivent avoir la même longueur")
        return self


# ------------------------------------------------------------------- phase 2
class LLMConfig(_Strict):
    backend: str = "ollama"
    url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b"
    keep_alive: str = "30m"
    num_ctx: int = Field(default=8192, gt=0)
    think: bool = False
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    max_tool_rounds: int = Field(default=6, ge=1)
    history_messages: int = Field(default=16, ge=2)


class Modulated(_Strict):
    """valeur = base + Σ poids × source, bornée à range."""
    base: float
    weights: dict[str, float] = {}
    range: tuple[float, float]


class ExpressionConfig(_Strict):
    temperature: Modulated
    max_tokens: Modulated
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    repeat_penalty: float = Field(default=1.1, gt=0.0)
    energy_cost_per_call: float = Field(default=0.004, ge=0.0)
    intensity_words: list[IntensityWord] = []
    urge_words: dict[str, str] = {}
    urge_threshold: float = 0.35


class Identity(_Strict):
    person: str | None = None
    name: str = "inconnu"
    role: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    minor: bool = False


class StimulusRef(_Strict):
    stimulus: str
    scale: float = 1.0


class AgentConfig(_Strict):
    console_identity: Identity
    tool_success: StimulusRef
    tool_failure: StimulusRef


class PermissionsConfig(_Strict):
    elevated_min_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    dangerous_min_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    owner_roles: list[str] = ["owner"]


class AppraisalRule(_Strict):
    stimulus: str
    scale: float = 1.0
    words: list[str]


class AppraisalFastConfig(_Strict):
    rules: list[AppraisalRule] = []


class PrinterConfig(_Strict):
    moonraker_url: str = "http://192.168.1.4"
    timeout_seconds: float = Field(default=4.0, gt=0.0)
    cache_seconds: float = Field(default=30.0, ge=0.0)
    cancel_macro: str | None = "CANCEL_PRINT"


# ------------------------------------------------------------------- vigie d'impression
class CameraSpec(_Strict):
    name: str
    snapshot_url: str                     # absolu, ou relatif à l'adresse de l'imprimante
    enabled: bool = True
    rotate: int = Field(default=0, ge=0, le=270)


class FrameQualitySpec(_Strict):
    dark: float = Field(default=25.0, ge=0.0)       # luminosité moyenne minimale (0-255)
    blur: float = Field(default=40.0, ge=0.0)       # netteté minimale (variance du laplacien)
    frozen_frames: int = Field(default=6, ge=2)     # images identiques d'affilée = caméra figée


class DetectorSpec(_Strict):
    model: str = "data/models/obico/model-weights-5a6b1be1fa.onnx"
    threshold: float = Field(default=0.08, gt=0.0, lt=1.0)     # Obico : 0,08
    nms: float = Field(default=0.45, gt=0.0, lt=1.0)
    device: str = "cpu"


class PredictorConfig(_Strict):
    """Constantes d'Obico (vérifiées dans leur code), voir printwatch/predict.py."""
    ewm_span: float = Field(default=12.0, gt=1.0)
    rolling_win_short: int = Field(default=310, ge=1)
    rolling_win_long: int = Field(default=7200, ge=1)
    init_safe_frames: int = Field(default=30, ge=0)
    threshold_low: float = 0.38
    threshold_high: float = 0.78
    short_multiple: float = 3.8
    escalating_factor: float = Field(default=1.75, ge=1.0)
    sensitivity: float = Field(default=1.0, ge=0.8, le=1.2)


class TelemetrySpec(_Strict):
    temp_tolerance: float = Field(default=8.0, gt=0.0)          # °C d'écart à la consigne
    temp_persist_seconds: float = Field(default=30.0, ge=0.0)
    stall_seconds: float = Field(default=900.0, gt=0.0)         # progression figée


class AutonomySpec(_Strict):
    max_level: int = Field(default=1, ge=0, le=1)   # 0 observe et alerte, 1 pause tout seul
    target: float = Field(default=0.98, gt=0.5, lt=1.0)
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    lead_min_seconds: float = Field(default=30.0, ge=0.0)
    max_false_pause_per_100h: float = Field(default=1.0, ge=0.0)
    min_ok_hours: float = Field(default=300.0, ge=0.0)


class PrintWatchConfig(_Strict):
    enabled: bool = True
    db: str = "vigie.db"
    frames_dir: str = "vigie"
    cameras: list[CameraSpec] = []
    discover_webcams: bool = True
    interval_seconds: float = Field(default=10.0, ge=1.0)
    quality: FrameQualitySpec = FrameQualitySpec()
    detector: DetectorSpec = DetectorSpec()
    predictor: PredictorConfig = PredictorConfig()
    telemetry: TelemetrySpec = TelemetrySpec()
    keep_frame_every_seconds: float = Field(default=60.0, ge=0.0)
    keep_days: float = Field(default=30.0, ge=1.0)
    triage: bool = True                    # Gemma regarde l'image quand la vigie s'inquiète
    autonomy: AutonomySpec = AutonomySpec()


class AtelierConfig(_Strict):
    stock_db: str = "atelier.db"
    checklist: str = "checklist.json"
    reminders: str = "rappels.json"
    pinouts: str = "pinouts.json"
    memory_db: str = "memoire.db"


# ----------------------------------------------------------------------------- voix
class VoiceCharacter(_Strict):
    """Chaîne d'effets de RAUB (profil « megatron »), reproduite à l'identique (avenant 3 §5)."""
    am_hz: float = Field(default=1.2, ge=0.0)          # modulation d'amplitude lente
    am_depth: float = Field(default=0.02, ge=0.0, le=0.5)
    wobble_hz: float = Field(default=0.0, ge=0.0)
    wobble_depth: float = Field(default=0.0, ge=0.0, le=0.5)
    nasal_gain_db: float = 0.0                         # pic à 1,7 kHz
    pitch_shift: float = -2.0    # nom de RAUB ; agit en fait comme un passe-bas à fs/4
    high_shelf_hz: float = Field(default=400.0, gt=0.0)
    high_shelf_db: float = -10.0
    echo_ms: float = Field(default=0.0, ge=0.0)
    echo_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    lo_fi_bits: int = Field(default=15, ge=2, le=16)
    drive: float = Field(default=1.4, gt=0.0)          # saturation tanh
    peak: float = Field(default=0.89, gt=0.0, le=1.0)


class XttsConfig(_Strict):
    model_dir: str = "data/models/xtts-v2"
    reference: str = "data/voice/xtts_ref.wav"
    language: str = "fr"
    device: str = "cuda"


class VoiceConfig(_Strict):
    enabled: bool = True
    backend: str = "xtts"
    xtts: XttsConfig = XttsConfig()
    character: VoiceCharacter = VoiceCharacter()
    max_sentence_chars: int = Field(default=230, ge=40, le=270)   # XTTS : 273 car. max en français


class ValdarConfig(_Strict):
    heart: HeartConfig
    temperament: TemperamentSpec
    initiative: InitiativeConfig
    memory: MemoryConfig = MemoryConfig()
    storage: StorageConfig = StorageConfig()
    sim: SimConfig
    llm: LLMConfig = LLMConfig()
    expression: ExpressionConfig | None = None
    agent: AgentConfig | None = None
    permissions: PermissionsConfig = PermissionsConfig()
    appraisal_fast: AppraisalFastConfig = AppraisalFastConfig()
    printer: PrinterConfig = PrinterConfig()
    atelier: AtelierConfig = AtelierConfig()
    voice: VoiceConfig = VoiceConfig()
    episodic: EpisodicConfig = EpisodicConfig()
    printwatch: PrintWatchConfig = PrintWatchConfig()
    thoughts: ThoughtsConfig = ThoughtsConfig()
    knowledge_db: str = "connaissances.db"

    _root: Path = PrivateAttr(default_factory=Path.cwd)

    @model_validator(mode="after")
    def _cross_check(self) -> ValdarConfig:
        for ev in self.sim.event_types:
            if ev not in self.heart.stimuli:
                raise ValueError(f"sim : stimulus inconnu '{ev}'")
        ign = self.initiative.ignored_stimulus
        if ign and ign not in self.heart.stimuli:
            raise ValueError(f"initiative : stimulus inconnu '{ign}'")
        refs = [r.stimulus for r in self.appraisal_fast.rules]
        if self.agent is not None:
            refs += [self.agent.tool_success.stimulus, self.agent.tool_failure.stimulus]
        for name in refs:
            if name not in self.heart.stimuli:
                raise ValueError(f"stimulus inconnu '{name}'")
        r = self.episodic.reinstate
        for tgt in list(r.positive) + list(r.negative):
            if tgt not in self.heart.variables:
                raise ValueError(f"episodic.reinstate : variable inconnue '{tgt}'")
        for tgt in list(self.thoughts.calm) + list(self.thoughts.stir):
            if tgt not in self.heart.variables:
                raise ValueError(f"thoughts : variable inconnue '{tgt}'")
        return self

    @property
    def root(self) -> Path:
        """Racine du dépôt (dossier parent de config/)."""
        return self._root

    def storage_path(self, name: str) -> Path:
        d = Path(self.storage.dir)
        if not d.is_absolute():
            d = self._root / d
        return d / name

    def repo_path(self, rel: str) -> Path:
        """Chemin relatif à la racine du dépôt (ou absolu tel quel)."""
        p = Path(rel)
        return p if p.is_absolute() else self._root / p


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "valdar.yaml"


def load(path: str | Path | None = None) -> ValdarConfig:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = ValdarConfig.model_validate(raw)
    cfg._root = p.resolve().parent.parent
    return cfg
