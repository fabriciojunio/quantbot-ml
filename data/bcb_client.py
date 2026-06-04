"""
Cliente do Banco Central do Brasil — séries temporais (SGS).

API pública, gratuita e SEM chave. É a fonte oficial para macroeconomia
brasileira com histórico longo (Selic, IPCA, CDI, câmbio, etc.). Perfeita
para o "passado e atual" que alimenta projeções de renda passiva.

Endpoint:
    https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados
        ?formato=json&dataInicial=DD/MM/AAAA&dataFinal=DD/MM/AAAA

Catálogo completo de séries: https://www3.bcb.gov.br/sgspub/

Todos os códigos abaixo foram testados ao vivo (scripts/_probe_sources.py).

Uso:
    bcb = BCBClient()
    print(bcb.selic_meta())          # 14.5
    print(bcb.cdi_anual())           # 14.4
    print(bcb.ipca_12m())            # inflação acumulada 12m (%)
    hist = bcb.historico("SELIC_META", anos=10)   # série histórica
"""

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("quantbot.data.bcb")

_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

# Séries do SGS mais úteis para renda passiva (código -> descrição).
SERIES: Dict[str, Tuple[int, str]] = {
    "SELIC_META":   (432,   "Meta Selic definida pelo Copom (% a.a.)"),
    "SELIC_DIARIA": (11,    "Taxa Selic efetiva diária (% a.d.)"),
    "CDI_ANUAL":    (4389,  "Taxa CDI anualizada (% a.a.)"),
    "CDI_DIARIO":   (12,    "Taxa CDI diária (% a.d.)"),
    "IPCA_MES":     (433,   "IPCA mensal (% no mês)"),
    "IGPM_MES":     (189,   "IGP-M mensal (% no mês)"),
    "USD_BRL":      (1,     "Dólar PTAX venda (R$)"),
    "EUR_BRL":      (21619, "Euro venda (R$)"),
    "POUPANCA":     (196,   "Rendimento da poupança (% a.m.)"),
    "PIB_MENSAL":   (4380,  "PIB mensal a preços de mercado (R$ milhões)"),
}


@dataclass
class Observacao:
    """Um ponto de uma série temporal."""
    data: date
    valor: float


class BCBClient:
    """
    Cliente HTTP para o SGS do Banco Central.

    Sem dependências externas (só urllib) e com cache em memória por sessão.
    Falha graciosamente: retorna None/lista vazia e loga, nunca crasha.
    """

    def __init__(self, timeout_seconds: int = 15):
        self.timeout = timeout_seconds
        self._cache: Dict[str, List[Observacao]] = {}

    # ── Atalhos de alto nível ───────────────────────────────────────────

    def selic_meta(self) -> Optional[float]:
        """Meta Selic atual (% a.a.) — ex.: 14.5."""
        return self._ultimo("SELIC_META")

    def cdi_anual(self) -> Optional[float]:
        """CDI anualizado atual (% a.a.)."""
        return self._ultimo("CDI_ANUAL")

    def usd_brl(self) -> Optional[float]:
        """Dólar PTAX de venda mais recente (R$)."""
        return self._ultimo("USD_BRL")

    def ipca_12m(self) -> Optional[float]:
        """IPCA acumulado dos últimos 12 meses (%)."""
        obs = self.historico("IPCA_MES", anos=2)
        if len(obs) < 12:
            return None
        acum = 1.0
        for o in obs[-12:]:
            acum *= (1 + o.valor / 100)
        return round((acum - 1) * 100, 2)

    # ── Núcleo ──────────────────────────────────────────────────────────

    def historico(
        self,
        serie: str,
        anos: int = 5,
        data_inicial: Optional[date] = None,
        data_final: Optional[date] = None,
    ) -> List[Observacao]:
        """
        Série histórica de uma das chaves de SERIES (ou um código int).

        Args:
            serie: chave de SERIES ("SELIC_META") ou o código numérico (432).
            anos: janela a partir de hoje, se data_inicial não for passada.
        """
        code = self._code(serie)
        if code is None:
            logger.warning(f"Série desconhecida: {serie}")
            return []

        if data_final is None:
            data_final = date.today()
        if data_inicial is None:
            data_inicial = data_final - timedelta(days=365 * anos + 5)

        cache_key = f"{code}:{data_inicial}:{data_final}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = (
            f"?formato=json"
            f"&dataInicial={data_inicial.strftime('%d/%m/%Y')}"
            f"&dataFinal={data_final.strftime('%d/%m/%Y')}"
        )
        url = _BASE.format(code=code) + params
        raw = self._get(url)
        obs: List[Observacao] = []
        for item in raw or []:
            try:
                d = datetime.strptime(item["data"], "%d/%m/%Y").date()
                obs.append(Observacao(data=d, valor=float(item["valor"])))
            except (KeyError, ValueError):
                continue
        self._cache[cache_key] = obs
        return obs

    def ultimo_valor(self, serie: str) -> Optional[Observacao]:
        """Última observação disponível de uma série."""
        code = self._code(serie)
        if code is None:
            return None
        url = _BASE.format(code=code) + "/ultimos/1?formato=json"
        raw = self._get(url)
        if not raw:
            return None
        try:
            item = raw[-1]
            d = datetime.strptime(item["data"], "%d/%m/%Y").date()
            return Observacao(data=d, valor=float(item["valor"]))
        except (KeyError, ValueError, IndexError):
            return None

    def snapshot(self) -> Dict[str, Optional[float]]:
        """Foto atual do macro: Selic, CDI, IPCA 12m, dólar."""
        return {
            "selic_meta": self.selic_meta(),
            "cdi_anual": self.cdi_anual(),
            "ipca_12m": self.ipca_12m(),
            "usd_brl": self.usd_brl(),
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    def _ultimo(self, serie: str) -> Optional[float]:
        o = self.ultimo_valor(serie)
        return o.valor if o else None

    @staticmethod
    def _code(serie) -> Optional[int]:
        if isinstance(serie, int):
            return serie
        if serie in SERIES:
            return SERIES[serie][0]
        if str(serie).isdigit():
            return int(serie)
        return None

    def _get(self, url: str) -> Optional[list]:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "QuantBot-ML/2.0 (research)"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            logger.warning(f"BCB HTTP {e.code} em {url}")
        except (urllib.error.URLError, TimeoutError) as e:
            logger.warning(f"BCB falha de rede: {e}")
        except json.JSONDecodeError:
            logger.warning("BCB retornou JSON inválido")
        return None
