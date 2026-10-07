"""La porte des 98 % : Valdar ne pilote l'imprimante tout seul que quand il a **prouvé** qu'il
voit venir au moins 98 % des échecs avant qu'ils deviennent irrattrapables.

Mesure (voir docs/connaissances/detection_etat_art.md) :
- **par événement**, pas par image : une impression ratée = un échec à attraper ;
- un échec est « attrapé » si une alerte a sonné au moins `lead_min_seconds` **avant le point
  de non-retour** noté par Olivier (ou, à défaut, avant la fin de l'impression) ;
- la réussite se juge sur la **borne basse** de l'intervalle de Clopper-Pearson (unilatéral,
  95 %) : 98 % mesurés sur 10 impressions ne prouvent rien. Il faut ~149 échecs tous attrapés ;
- les fausses alertes comptent aussi : pauses injustifiées par heure d'impression réussie.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X ≥ k) pour X ~ Binomiale(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k, n + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                          + i * lp + (n - i) * lq)
    return min(1.0, total)


def clopper_pearson_lower(k: int, n: int, confidence: float = 0.95) -> float:
    """Borne basse unilatérale exacte de la proportion de succès (k succès sur n)."""
    if n <= 0 or k <= 0:
        return 0.0
    alpha = 1.0 - confidence
    if k == n:
        return alpha ** (1.0 / n)
    lo, hi = 0.0, k / n
    for _ in range(80):                 # P(X ≥ k | p) croît avec p : bisection
        mid = (lo + hi) / 2
        if binom_sf(k, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return lo


def events_needed(target: float = 0.98, confidence: float = 0.95, misses: int = 0) -> int:
    """Nombre d'échecs à observer (avec `misses` ratés) pour que la borne basse atteigne la cible."""
    n = misses + 1
    while clopper_pearson_lower(n - misses, n, confidence) < target:
        n += 1
    return n


@dataclass
class PrintRecord:
    job: str
    started: float
    ended: float
    failed: bool                       # étiquette (Olivier ou état Moonraker confirmé)
    no_return: float | None = None     # moment où l'échec devient irrattrapable
    alerts: list[float] = field(default_factory=list)
    pauses: list[float] = field(default_factory=list)   # pauses que Valdar aurait faites

    @property
    def hours(self) -> float:
        return max(0.0, self.ended - self.started) / 3600.0


@dataclass
class GateReport:
    failures: int
    caught: int
    recall: float
    recall_lower: float
    lead_median_s: float | None
    ok_prints: int
    ok_hours: float
    false_pauses: int
    false_pause_rate_per_100h: float | None
    needed_more: int
    open: bool
    reasons: list[str]

    def text(self) -> str:
        lines = [f"échecs observés : {self.failures}, vus à temps : {self.caught} "
                 f"({self.recall:.0%} mesuré, au moins {self.recall_lower:.1%} prouvé)"]
        if self.lead_median_s is not None:
            lines.append(f"avance médiane : {self.lead_median_s / 60:.1f} min avant le point "
                         "de non-retour")
        lines.append(f"impressions réussies : {self.ok_prints} ({self.ok_hours:.0f} h), "
                     f"fausses pauses : {self.false_pauses}")
        lines.append("autonomie : " + ("DÉBLOQUÉE" if self.open else "verrouillée — "
                                       + " ; ".join(self.reasons)))
        return "\n".join(lines)


def gate(records: list[PrintRecord], target: float = 0.98, confidence: float = 0.95,
         lead_min_seconds: float = 30.0, max_false_pause_per_100h: float = 1.0,
         min_ok_hours: float = 300.0) -> GateReport:
    fails = [r for r in records if r.failed]
    oks = [r for r in records if not r.failed]
    caught, leads = 0, []
    for r in fails:
        limit = r.no_return if r.no_return is not None else r.ended
        early = [a for a in r.alerts if a <= limit - lead_min_seconds and a >= r.started]
        if early:
            caught += 1
            leads.append(limit - min(early))
    n = len(fails)
    recall = caught / n if n else 0.0
    lower = clopper_pearson_lower(caught, n, confidence)
    ok_hours = sum(r.hours for r in oks)
    false_pauses = sum(len(r.pauses) for r in oks)
    rate = false_pauses / ok_hours * 100 if ok_hours > 0 else None
    reasons = []
    if lower < target:
        need = events_needed(target, confidence, misses=n - caught) - n
        reasons.append(f"il faut encore au moins {max(1, need)} échec(s) attrapé(s) à temps "
                       f"sans en rater (borne basse {lower:.1%} < {target:.0%})")
    else:
        need = 0
    if ok_hours < min_ok_hours:
        reasons.append(f"pas assez d'heures d'impressions réussies observées ({ok_hours:.0f} h "
                       f"sur {min_ok_hours:.0f} h) pour juger les fausses pauses")
    elif rate is not None and rate > max_false_pause_per_100h:
        reasons.append(f"trop de fausses pauses ({rate:.1f} pour 100 h)")
    leads.sort()
    return GateReport(n, caught, recall, lower, leads[len(leads) // 2] if leads else None,
                      len(oks), ok_hours, false_pauses, rate, need, not reasons, reasons)


def records_from_rows(rows: list[dict[str, Any]]) -> list[PrintRecord]:
    return [PrintRecord(job=r["job"], started=r["started"], ended=r["ended"],
                        failed=bool(r["failed"]), no_return=r.get("no_return"),
                        alerts=list(r.get("alerts", [])), pauses=list(r.get("pauses", [])))
            for r in rows if r.get("failed") is not None]
