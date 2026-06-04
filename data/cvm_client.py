"""
Cliente CVM — dados oficiais e públicos de Fundos Imobiliários.

Por que usar dados da CVM além da brapi.dev
---------------------------------------------
A brapi.dev é ótima para cotações e dados de mercado. Mas para análise
fundamentalista séria de FIIs, os dados definitivos vêm da CVM:

    1. Patrimônio Líquido oficial (PL mensal, obrigação regulatória)
    2. Número de cotistas (CVM exige divulgação mensal)
    3. Composição da carteira (imóveis, CRIs, outros ativos — CNPJ dos imóveis)
    4. Resultado contábil por cota (base para calcular sustentabilidade do DY)
    5. Vacância oficial (diferente da vacância estimada por analistas)

O Portal Dados Abertos CVM (dados.cvm.gov.br) disponibiliza:
    - Informes mensais de todos os FIIs: DESDE 2016
    - Cadastro atualizado: todo dia útil
    - Histórico completo e gratuito: sem autenticação

URLs base:
    Cadastro: https://dados.cvm.gov.br/dados/FII/CAD/DADOS/inf_cadastral_fii.csv
    Informe:  https://dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/
              inf_mensal_fii_AAAAMM.csv (ex: inf_mensal_fii_202605.csv)

Referência: CVM Instrução 571/2015, Anexo 39-I (Informe Mensal de FII).
"""

import csv
import io
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
import urllib.error
import json

logger = logging.getLogger("quantbot.data.cvm_client")

_BASE = "https://dados.cvm.gov.br/dados/FII"
_CADASTRO_URL = f"{_BASE}/CAD/DADOS/inf_cadastral_fii.csv"
_INFORME_URL = f"{_BASE}/DOC/INF_MENSAL/DADOS/inf_mensal_fii_{{aaaamm}}.csv"


@dataclass
class FIIRegistration:
    """Dados cadastrais oficiais de um FII (fonte: CVM)."""
    cnpj_fundo: str
    nome_fundo: str
    nome_administrador: str
    cnpj_administrador: str
    data_registro: str          # YYYY-MM-DD
    situacao: str               # "EM FUNCIONAMENTO NORMAL", "CANCELADO", etc.
    tipo: str                   # "FII" (todos), subtipo varia
    prazo_encerramento: str     # "INDETERMINADO" para a maioria
    sede_municipio: str
    sede_uf: str
    ticker: str = ""            # Preenchido via lookup manual


@dataclass
class FIIMonthlyReport:
    """Informe mensal oficial de um FII (fonte: CVM)."""
    cnpj_fundo: str
    competencia: str            # YYYY-MM
    pl: float                   # Patrimônio Líquido em R$
    n_cotistas: int             # Número de cotistas
    cota_valor: float           # Valor da cota no período
    rendimentos_pagos: float    # Total distribuído no mês
    rendimento_por_cota: float  # R$ por cota
    ativos_total: float         # Total de ativos
    passivos_total: float       # Total de passivos
    resultado_mes: float        # Resultado contábil do mês


class CVMClient:
    """
    Acessa dados públicos da CVM para FIIs.

    Todos os endpoints são públicos e gratuitos — dados oficiais,
    obrigatoriamente reportados pelos gestores dos fundos.

    Uso:
        cvm = CVMClient()

        # Busca cadastro de todos os FIIs ativos
        fiis = cvm.get_fii_registration(apenas_ativos=True)

        # Busca informe mensal recente
        relatorios = cvm.get_monthly_reports(year=2026, month=5)

        # Dados consolidados de um FII específico
        dados = cvm.get_fii_summary("17.022.084/0001-53")  # HGLG11
    """

    def __init__(
        self,
        cache_dir: str = "data/cache/cvm",
        timeout: int = 30,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    # ── Cadastro ─────────────────────────────────────────────────────────────

    def get_fii_registration(
        self,
        apenas_ativos: bool = True,
        cache_hours: int = 24,
    ) -> List[FIIRegistration]:
        """
        Retorna cadastro de todos os FIIs registrados na CVM.

        Args:
            apenas_ativos: filtra apenas fundos EM FUNCIONAMENTO NORMAL
            cache_hours: horas de cache local (atualizado diariamente pela CVM)

        Returns:
            Lista de FIIRegistration com dados cadastrais oficiais.
        """
        cache_file = self.cache_dir / "cadastro_fii.json"
        cached = self._load_cache(cache_file, max_age_hours=cache_hours)
        if cached is not None:
            registrations = [FIIRegistration(**r) for r in cached]
            if apenas_ativos:
                return [r for r in registrations if "FUNCIONAMENTO" in r.situacao.upper()]
            return registrations

        raw_csv = self._fetch_url(_CADASTRO_URL)
        if not raw_csv:
            logger.warning("CVM: falha ao baixar cadastro de FIIs. Usando cache ou offline.")
            return []

        registrations = self._parse_cadastro_csv(raw_csv)

        # Salva cache
        self._save_cache(cache_file, [r.__dict__ for r in registrations])

        if apenas_ativos:
            return [r for r in registrations if "FUNCIONAMENTO" in r.situacao.upper()]
        return registrations

    def get_monthly_reports(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
        cache_hours: int = 48,
    ) -> List[FIIMonthlyReport]:
        """
        Retorna informes mensais de todos os FIIs de um período.

        Se year/month não especificados, usa o mês anterior ao atual.

        Args:
            year: ano (ex: 2026)
            month: mês (1-12)
            cache_hours: horas de cache

        Returns:
            Lista de FIIMonthlyReport com dados contábeis oficiais.
        """
        if year is None or month is None:
            today = date.today()
            if today.month == 1:
                year, month = today.year - 1, 12
            else:
                year, month = today.year, today.month - 1

        aaaamm = f"{year}{month:02d}"
        cache_file = self.cache_dir / f"informe_mensal_{aaaamm}.json"
        cached = self._load_cache(cache_file, max_age_hours=cache_hours)
        if cached is not None:
            return [FIIMonthlyReport(**r) for r in cached]

        url = _INFORME_URL.format(aaaamm=aaaamm)
        raw_csv = self._fetch_url(url)
        if not raw_csv:
            logger.warning(f"CVM: informe mensal {aaaamm} não disponível.")
            return []

        reports = self._parse_informe_csv(raw_csv)
        self._save_cache(cache_file, [r.__dict__ for r in reports])
        return reports

    def get_fii_summary(
        self,
        cnpj: str,
        months_back: int = 3,
    ) -> Optional[Dict]:
        """
        Dados consolidados de um FII: cadastro + últimos informes mensais.

        Útil para validar dados da brapi.dev com a fonte oficial.
        """
        registrations = self.get_fii_registration()
        cadastro = next((r for r in registrations if r.cnpj_fundo == cnpj), None)

        relatorios = []
        today = date.today()
        for i in range(months_back):
            m = today.month - i - 1
            y = today.year
            while m <= 0:
                m += 12
                y -= 1
            reps = self.get_monthly_reports(year=y, month=m)
            monthly = [r for r in reps if r.cnpj_fundo == cnpj]
            if monthly:
                relatorios.append(monthly[0])

        if not cadastro and not relatorios:
            return None

        result = {
            "cnpj": cnpj,
            "cadastro": cadastro.__dict__ if cadastro else None,
            "ultimos_informes": [r.__dict__ for r in relatorios],
        }

        if relatorios:
            ultimo = relatorios[0]
            result["pl_atual"] = ultimo.pl
            result["n_cotistas"] = ultimo.n_cotistas
            result["rendimento_por_cota"] = ultimo.rendimento_por_cota
            result["resultado_mes"] = ultimo.resultado_mes

        return result

    def cross_validate_brapi(
        self,
        ticker: str,
        brapi_provento: float,
        cnpj: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> Dict:
        """
        Valida dados da brapi.dev contra a fonte oficial CVM.

        Detecta discrepâncias entre o provento reportado na API
        e o valor oficial do informe mensal da CVM.
        """
        if year is None or month is None:
            today = date.today()
            year = today.year
            month = today.month - 1 if today.month > 1 else 12

        reports = self.get_monthly_reports(year=year, month=month)
        cvm_report = next((r for r in reports if r.cnpj_fundo == cnpj), None)

        if not cvm_report:
            return {
                "ticker": ticker,
                "validado": False,
                "motivo": "CNPJ não encontrado no informe CVM do período",
            }

        diff_pct = abs(brapi_provento - cvm_report.rendimento_por_cota)
        diff_ok = diff_pct < 0.05  # tolerância de R$0.05

        return {
            "ticker": ticker,
            "validado": diff_ok,
            "brapi_valor": brapi_provento,
            "cvm_valor": cvm_report.rendimento_por_cota,
            "diferenca": round(brapi_provento - cvm_report.rendimento_por_cota, 4),
            "pl_oficial": cvm_report.pl,
            "n_cotistas_oficial": cvm_report.n_cotistas,
        }

    # ── Parsing CSV ──────────────────────────────────────────────────────────

    def _parse_cadastro_csv(self, raw: str) -> List[FIIRegistration]:
        """Parseia o CSV de cadastro de FIIs da CVM."""
        registrations = []
        try:
            reader = csv.DictReader(io.StringIO(raw), delimiter=";")
            for row in reader:
                try:
                    reg = FIIRegistration(
                        cnpj_fundo=self._clean(row.get("CNPJ_FUNDO", "")),
                        nome_fundo=self._clean(row.get("DENOM_SOCIAL", "")),
                        nome_administrador=self._clean(row.get("NOME_ADMIN", "")),
                        cnpj_administrador=self._clean(row.get("CNPJ_ADMIN", "")),
                        data_registro=self._clean(row.get("DT_REG", "")),
                        situacao=self._clean(row.get("SIT", "")),
                        tipo=self._clean(row.get("TP_FUNDO", "FII")),
                        prazo_encerramento=self._clean(row.get("PRAZO", "INDETERMINADO")),
                        sede_municipio=self._clean(row.get("MUNI_GESTOR", "")),
                        sede_uf=self._clean(row.get("UF_GESTOR", "")),
                    )
                    registrations.append(reg)
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"CVM: erro ao parsear cadastro CSV: {e}")
        return registrations

    def _parse_informe_csv(self, raw: str) -> List[FIIMonthlyReport]:
        """Parseia o CSV de informe mensal de FIIs."""
        reports = []
        try:
            reader = csv.DictReader(io.StringIO(raw), delimiter=";")
            for row in reader:
                try:
                    report = FIIMonthlyReport(
                        cnpj_fundo=self._clean(row.get("CNPJ_FUNDO", "")),
                        competencia=self._clean(row.get("COMPETENCIA", "")),
                        pl=self._to_float(row.get("VL_PATRIM_LIQ", "0")),
                        n_cotistas=self._to_int(row.get("NR_COTST", "0")),
                        cota_valor=self._to_float(row.get("VL_COTA", "0")),
                        rendimentos_pagos=self._to_float(row.get("VL_RENDTO_PAGO", "0")),
                        rendimento_por_cota=self._to_float(row.get("VL_RENDTO_PAGO_COTA", "0")),
                        ativos_total=self._to_float(row.get("VL_TOTAL_ATIVOS", "0")),
                        passivos_total=self._to_float(row.get("VL_TOTAL_PASSIVOS", "0")),
                        resultado_mes=self._to_float(row.get("VL_RESULT_MES", "0")),
                    )
                    if report.cnpj_fundo:
                        reports.append(report)
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"CVM: erro ao parsear informe CSV: {e}")
        return reports

    # ── Utilitários ───────────────────────────────────────────────────────────

    def _fetch_url(self, url: str) -> Optional[str]:
        """Faz o download de um CSV público da CVM."""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "QuantBot/2.0 (CVM open data; educational)"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                # CVM usa ISO-8859-1 (latin-1)
                return raw.decode("iso-8859-1", errors="replace")
        except Exception as e:
            logger.debug(f"CVM fetch falhou para {url}: {e}")
            return None

    @staticmethod
    def _clean(value: str) -> str:
        return (value or "").strip()

    @staticmethod
    def _to_float(value: str) -> float:
        """Converte string numérica BR (ex: '3.500.000,50') para float."""
        try:
            v = (value or "0").strip()
            # Formato BR: pontos como separador de milhar, vírgula como decimal
            # Ex: "3.500.000,50" → 3500000.50
            if "," in v:
                v = v.replace(".", "").replace(",", ".")
            return float(v)
        except (ValueError, AttributeError):
            return 0.0

    @staticmethod
    def _to_int(value: str) -> int:
        try:
            return int((value or "0").strip())
        except (ValueError, AttributeError):
            return 0

    def _load_cache(self, path: Path, max_age_hours: int) -> Optional[List]:
        if not path.exists():
            return None
        try:
            stat = path.stat()
            age_hours = (datetime.now().timestamp() - stat.st_mtime) / 3600
            if age_hours > max_age_hours:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_cache(self, path: Path, data: list) -> None:
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning(f"CVM: falha ao salvar cache {path}: {e}")
