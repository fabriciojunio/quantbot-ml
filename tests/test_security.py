"""Testes para o módulo de segurança."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.security import (
    PasswordManager, EncryptionManager, AuditTrail,
    LGPDCompliance, RateLimiter, SecurityManager,
)


class TestPasswordManager:

    def test_setup_and_verify(self, tmp_path):
        pm = PasswordManager(str(tmp_path / ".cred"))
        pm.setup_password("Teste@123!")
        assert pm.verify_password("Teste@123!")
        assert not pm.verify_password("errada")

    def test_weak_password_rejected(self, tmp_path):
        pm = PasswordManager(str(tmp_path / ".cred"))
        with pytest.raises(ValueError, match="8 caracteres"):
            pm.setup_password("abc")

    def test_no_uppercase_rejected(self, tmp_path):
        pm = PasswordManager(str(tmp_path / ".cred"))
        with pytest.raises(ValueError, match="maiúscula"):
            pm.setup_password("teste@123!")

    def test_no_number_rejected(self, tmp_path):
        pm = PasswordManager(str(tmp_path / ".cred"))
        with pytest.raises(ValueError, match="número"):
            pm.setup_password("Teste@abc!")

    def test_no_special_rejected(self, tmp_path):
        pm = PasswordManager(str(tmp_path / ".cred"))
        with pytest.raises(ValueError, match="especial"):
            pm.setup_password("Teste1234")

    def test_persistence(self, tmp_path):
        cred_path = str(tmp_path / ".cred")
        pm1 = PasswordManager(cred_path)
        pm1.setup_password("Forte@123!")
        pm2 = PasswordManager(cred_path)
        assert pm2.verify_password("Forte@123!")

    def test_is_configured(self, tmp_path):
        pm = PasswordManager(str(tmp_path / ".cred"))
        assert not pm.is_configured()
        pm.setup_password("Teste@123!")
        assert pm.is_configured()


class TestEncryption:

    def test_encrypt_decrypt(self):
        enc = EncryptionManager(master_key="test_key_123")
        plaintext = "minha_api_key_secreta_12345"
        encrypted = enc.encrypt(plaintext)
        assert encrypted != plaintext
        decrypted = enc.decrypt(encrypted)
        assert decrypted == plaintext

    def test_different_keys_fail(self):
        enc1 = EncryptionManager(master_key="key1")
        enc2 = EncryptionManager(master_key="key2")
        encrypted = enc1.encrypt("dados secretos")
        with pytest.raises(Exception):
            enc2.decrypt(encrypted)

    def test_tampered_data_detected(self):
        enc = EncryptionManager(master_key="test_key")
        encrypted = enc.encrypt("dados")
        tampered = encrypted[:-5] + "XXXXX"
        with pytest.raises(Exception):
            enc.decrypt(tampered)


class TestAuditTrail:

    def test_log_entry(self, tmp_path):
        audit = AuditTrail(str(tmp_path / "audit.json"))
        entry = audit.log("test_action", "detalhes do teste")
        assert entry.action == "test_action"
        assert entry.checksum != ""

    def test_integrity_check(self, tmp_path):
        audit = AuditTrail(str(tmp_path / "audit.json"))
        audit.log("action1", "details1")
        audit.log("action2", "details2")
        audit.log("action3", "details3")
        is_valid, msg = audit.verify_integrity()
        assert is_valid is True

    def test_tampered_entry_detected(self, tmp_path):
        audit = AuditTrail(str(tmp_path / "audit.json"))
        audit.log("action1", "details1")
        audit.log("action2", "details2")
        # Adultera uma entrada
        audit.entries[0].details = "adulterado!"
        is_valid, msg = audit.verify_integrity()
        assert is_valid is False
        assert "adulterada" in msg.lower() or "inválido" in msg.lower()

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "audit.json")
        audit1 = AuditTrail(path)
        audit1.log("test", "persistent")
        audit2 = AuditTrail(path)
        assert len(audit2.entries) == 1


class TestLGPD:

    def test_register_consent(self, tmp_path):
        lgpd = LGPDCompliance(str(tmp_path / "lgpd"))
        record = lgpd.register_consent("user1", "trading", ["config", "orders"], True)
        assert record["consent_given"] is True
        assert record["legal_basis"] == "consent"

    def test_revoke_consent(self, tmp_path):
        lgpd = LGPDCompliance(str(tmp_path / "lgpd"))
        lgpd.register_consent("user1", "trading", ["config"], True)
        result = lgpd.revoke_consent("user1", "trading")
        assert result is True

    def test_get_user_data(self, tmp_path):
        lgpd = LGPDCompliance(str(tmp_path / "lgpd"))
        lgpd.register_consent("user1", "trading", ["config"], True)
        data = lgpd.get_user_data("user1")
        assert data["user_id"] == "user1"
        assert len(data["consents"]) == 1

    def test_delete_user_data(self, tmp_path):
        lgpd = LGPDCompliance(str(tmp_path / "lgpd"))
        lgpd.register_consent("user1", "trading", ["config"], True)
        result = lgpd.delete_user_data("user1")
        assert result is True

    def test_export_user_data(self, tmp_path):
        lgpd = LGPDCompliance(str(tmp_path / "lgpd"))
        lgpd.register_consent("user1", "trading", ["config"], True)
        export = lgpd.export_user_data("user1")
        assert export["portable"] is True
        assert export["export_format"] == "JSON"

    def test_privacy_policy(self, tmp_path):
        lgpd = LGPDCompliance(str(tmp_path / "lgpd"))
        policy = lgpd.get_privacy_policy()
        assert "LGPD" in policy
        assert "Art. 18" in policy

    def test_breach_report(self, tmp_path):
        lgpd = LGPDCompliance(str(tmp_path / "lgpd"))
        report = lgpd.report_breach("Vazamento de dados", 50, ["email"])
        assert report["status"] == "detected"
        assert "deadline_anpd" in report


class TestRateLimiter:

    def test_allows_within_limit(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            allowed, _ = rl.check("test")
            assert allowed is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.check("test")
        rl.check("test")
        allowed, reason = rl.check("test")
        assert allowed is False
        assert "Rate limit" in reason


class TestSecurityManager:

    def test_full_flow(self, tmp_path):
        os.chdir(tmp_path)
        sec = SecurityManager()
        sec.setup_password("Forte@123!")
        assert sec.authenticate("Forte@123!")
        assert not sec.authenticate("errada")

    def test_encrypt_api_key(self, tmp_path):
        os.chdir(tmp_path)
        sec = SecurityManager()
        encrypted = sec.encrypt_api_key("sk-12345-secret")
        decrypted = sec.decrypt_api_key(encrypted)
        assert decrypted == "sk-12345-secret"

    def test_security_status(self, tmp_path):
        os.chdir(tmp_path)
        sec = SecurityManager()
        status = sec.get_security_status()
        assert "password_configured" in status
        assert "audit_trail_integrity" in status
        assert "lgpd_compliant" in status
