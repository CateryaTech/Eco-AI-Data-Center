"""
blockchain_integration.py
==========================
Blockchain + ZK-proof integration untuk Eco AI Data Center.

Features:
  • ZKCommitment  — privacy-preserving commitment (Merkle + HMAC)
  • IPFSPublisher — publish audit records ke Pinata/IPFS lokal/mock
  • PolygonPublisher — publish commitment hash ke Polygon (mock jika web3 tidak ada)
  • ESGBadge      — CATERYA-Verified NFT-like token (ERC-721 compatible)
  • BlockchainIntegration — orchestrator utama

Zero-trust: hanya commitment hash yang on-chain; raw data tidak pernah keluar.

Eco AI Data Center — CateryaTech
Author  : Ary HH | cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import hashlib, hmac as _hmac, json, logging, os, secrets, time, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from provenance import ProvenanceChain

logger = logging.getLogger("eco_ai.blockchain")

# ── optional deps ─────────────────────────────────────────────────────────
try:
    from web3 import Web3                                    # type: ignore
    # geth_poa_middleware was renamed/moved in web3 v6+; try both paths
    try:
        from web3.middleware import geth_poa_middleware      # type: ignore  # web3 <6
    except ImportError:
        try:
            from web3.middleware import ExtraDataToPOAMiddleware as geth_poa_middleware  # type: ignore  # web3 >=6
        except ImportError:
            geth_poa_middleware = None                       # type: ignore
    WEB3_AVAILABLE = True
except ImportError:
    Web3 = None; WEB3_AVAILABLE = False

try:
    import requests as _req; REQUESTS_AVAILABLE = True
except ImportError:
    _req = None; REQUESTS_AVAILABLE = False

POLYGON_MUMBAI_RPC  = "https://rpc-mumbai.maticvigil.com"
POLYGON_MAINNET_RPC = "https://polygon-rpc.com"
IPFS_GATEWAY        = "https://ipfs.io/ipfs/"
PINATA_API_URL      = "https://api.pinata.cloud/pinning/pinJSONToIPFS"


# ══════════════════════════════════════════════════════════════════════════
# ZK Commitment — privacy layer
# ══════════════════════════════════════════════════════════════════════════

class ZKCommitment:
    """
    Pedersen-style commitment: C = HMAC-SHA256(blinding_factor, data).
    Only C goes on-chain. Raw data stays off-chain.
    Verify anytime with data + blinding_factor.
    """

    _PII = {
        "email","name","user_id","username","phone","address",
        "npwp","nik","passport","account_no","rekening","ip_address",
    }

    @classmethod
    def commit(cls, data: Dict[str, Any],
               blinding_factor: Optional[str] = None) -> Dict[str, str]:
        """Return commitment dict. Keep blinding_factor private."""
        bf = blinding_factor or secrets.token_hex(32)
        clean = cls._scrub(data)
        raw   = json.dumps(clean, sort_keys=True).encode()
        commitment  = _hmac.new(bf.encode(), raw, hashlib.sha256).hexdigest()
        data_hash   = hashlib.sha256(raw).hexdigest()
        merkle_root = cls._merkle(clean, bf)
        return {
            "commitment":      commitment,
            "blinding_factor": bf,
            "data_hash":       data_hash,
            "merkle_root":     merkle_root,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def verify(cls, data: Dict[str, Any],
               commitment: str, blinding_factor: str) -> bool:
        """Verify commitment given raw data + blinding factor."""
        clean = cls._scrub(data)
        raw   = json.dumps(clean, sort_keys=True).encode()
        expected = _hmac.new(blinding_factor.encode(), raw, hashlib.sha256).hexdigest()
        return _hmac.compare_digest(commitment, expected)

    @classmethod
    def _scrub(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in data.items()
                if k.lower() not in cls._PII
                and isinstance(v, (int, float, bool, str, type(None)))}

    @classmethod
    def _merkle(cls, data: Dict[str, Any], bf: str) -> str:
        leaves = [
            hashlib.sha256(f"{k}:{v}:{bf[:8]}".encode()).hexdigest()
            for k, v in sorted(data.items())
        ]
        if not leaves:
            return hashlib.sha256(b"empty").hexdigest()
        while len(leaves) > 1:
            if len(leaves) % 2:
                leaves.append(leaves[-1])
            leaves = [
                hashlib.sha256((leaves[i] + leaves[i+1]).encode()).hexdigest()
                for i in range(0, len(leaves), 2)
            ]
        return leaves[0]


# ══════════════════════════════════════════════════════════════════════════
# ESG Badge
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ESGBadge:
    """CATERYA-Verified ESG Badge — ERC-721 compatible metadata."""
    badge_id:          str
    recipient:         str
    badge_type:        str
    cos_score:         float
    pue_score:         float
    compliance_status: str
    commitment_hash:   str
    merkle_root:       str
    issuer:            str   = "CateryaTech"
    issuer_did:        str   = "did:polygon:cateryatech"
    chain:             str   = "polygon-mumbai"
    token_id:          Optional[str] = None
    tx_hash:           Optional[str] = None
    ipfs_cid:          Optional[str] = None
    issued_at:         str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata_uri:      Optional[str] = None
    signature:         Optional[str] = None

    def to_metadata_dict(self) -> dict:
        return {
            "name": f"CATERYA-Verified ESG Badge — {self.badge_type}",
            "description": (
                f"This badge certifies {self.recipient} achieved ethical AI compliance "
                f"via the CATERYA framework. COS: {self.cos_score:.4f} | PUE: {self.pue_score:.4f}."
            ),
            "image":        f"ipfs://{self.ipfs_cid or 'QmCATERYABadge'}",
            "external_url": "https://cateryatech.com/verify",
            "attributes": [
                {"trait_type": "Badge Type",        "value": self.badge_type},
                {"trait_type": "COS Score",         "value": round(self.cos_score,4), "display_type":"number"},
                {"trait_type": "PUE",               "value": round(self.pue_score,4), "display_type":"number"},
                {"trait_type": "Issuer",            "value": self.issuer},
                {"trait_type": "Chain",             "value": self.chain},
                {"trait_type": "Compliance",        "value": self.compliance_status},
                {"trait_type": "Issued At",         "value": self.issued_at},
                {"trait_type": "Commitment",        "value": self.commitment_hash[:20]+"..."},
            ],
            "caterya": {
                "framework_version": "2.0.0",
                "commitment_hash":   self.commitment_hash,
                "merkle_root":       self.merkle_root,
                "issuer_did":        self.issuer_did,
                "token_id":          self.token_id,
                "tx_hash":           self.tx_hash,
            },
        }

    def as_verifiable_credential(self) -> dict:
        return {
            "@context": ["https://www.w3.org/2018/credentials/v1",
                         "https://cateryatech.com/contexts/esg-badge/v1"],
            "id":           f"https://cateryatech.com/badges/{self.badge_id}",
            "type":         ["VerifiableCredential", "CATERYAESGBadge"],
            "issuer":       self.issuer_did,
            "issuanceDate": self.issued_at,
            "credentialSubject": {
                "id": self.recipient, "badgeType": self.badge_type,
                "cosScore": self.cos_score, "pue": self.pue_score,
                "complianceStatus": self.compliance_status,
                "commitmentHash": self.commitment_hash,
                "merkleRoot": self.merkle_root,
                "chain": self.chain, "tokenId": self.token_id,
                "txHash": self.tx_hash, "ipfsCid": self.ipfs_cid,
            },
            "proof": {"type": "CATERYAHMACSignature2026",
                      "created": self.issued_at,
                      "value": self.signature or "unsigned"},
        }

    def verify_signature(self, signing_key: str) -> bool:
        if not self.signature:
            return False
        payload = f"{self.badge_id}:{self.commitment_hash}:{self.issued_at}"
        expected = _hmac.new(signing_key.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return _hmac.compare_digest(self.signature, expected)


# ══════════════════════════════════════════════════════════════════════════
# IPFS Publisher
# ══════════════════════════════════════════════════════════════════════════

class IPFSPublisher:
    """Publish JSON to IPFS via Pinata → local node → mock CID."""

    def __init__(self, pinata_api_key: Optional[str]=None,
                 pinata_secret: Optional[str]=None,
                 local_ipfs_url: str="http://localhost:5001"):
        self.pinata_api_key = pinata_api_key or os.getenv("PINATA_API_KEY")
        self.pinata_secret  = pinata_secret  or os.getenv("PINATA_SECRET")
        self.local_ipfs_url = local_ipfs_url

    def publish(self, data: dict, name: str = "caterya-record") -> str:
        raw_bytes = json.dumps(data, sort_keys=True, indent=2).encode()
        mock_cid  = "Qm" + hashlib.sha256(raw_bytes).hexdigest()[:44]

        if self.pinata_api_key and self.pinata_secret and REQUESTS_AVAILABLE:
            try:
                resp = _req.post(PINATA_API_URL,
                    json={"pinataMetadata": {"name": name}, "pinataContent": data},
                    headers={"pinata_api_key": self.pinata_api_key,
                             "pinata_secret_api_key": self.pinata_secret},
                    timeout=15)
                if resp.status_code == 200:
                    cid = resp.json()["IpfsHash"]
                    logger.info("[IPFS] Pinata: %s", cid)
                    return cid
            except Exception as e:
                logger.warning("[IPFS] Pinata failed: %s", e)

        if REQUESTS_AVAILABLE:
            try:
                resp = _req.post(f"{self.local_ipfs_url}/api/v0/add",
                    files={"file": ("data.json", raw_bytes, "application/json")},
                    timeout=5)
                if resp.status_code == 200:
                    cid = resp.json()["Hash"]
                    logger.info("[IPFS] Local node: %s", cid)
                    return cid
            except Exception:
                pass

        logger.info("[IPFS] Mock CID: %s", mock_cid)
        return mock_cid

    def gateway_url(self, cid: str) -> str:
        return f"{IPFS_GATEWAY}{cid}"


# ══════════════════════════════════════════════════════════════════════════
# Polygon Publisher
# ══════════════════════════════════════════════════════════════════════════

class PolygonPublisher:
    """
    Publish ZK commitment hashes ke Polygon blockchain.
    Jika web3 tidak tersedia atau wallet tidak dikonfigurasi → mock tx hash.
    Hanya commitment hash yang on-chain, BUKAN data mentah.
    """

    def __init__(self, rpc_url: Optional[str]=None,
                 contract_address: Optional[str]=None,
                 private_key: Optional[str]=None,
                 network: str="mumbai"):
        self.rpc_url          = rpc_url          or os.getenv("POLYGON_RPC_URL", POLYGON_MUMBAI_RPC)
        self.contract_address = contract_address or os.getenv("BADGE_CONTRACT_ADDR", "")
        self.private_key      = private_key      or os.getenv("POLYGON_PRIVATE_KEY", "")
        self.network          = network
        self._w3              = None

    def _connect(self):
        if not WEB3_AVAILABLE or self._w3:
            return self._w3
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        if geth_poa_middleware is not None:
            w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self._w3 = w3
        return w3

    def publish_commitment(self, commitment: str, metadata_uri: str="",
                           event_type: str="audit_trail") -> str:
        w3 = self._connect()
        if not w3 or not w3.is_connected() or not self.private_key:
            return self._mock_tx(commitment)
        try:
            acct  = w3.eth.account.from_key(self.private_key)
            nonce = w3.eth.get_transaction_count(acct.address)
            data  = bytes.fromhex(commitment[:64].ljust(64,"0"))
            tx = {"from": acct.address, "to": acct.address, "value": 0,
                  "gas": 50000, "gasPrice": w3.eth.gas_price, "nonce": nonce,
                  "data": w3.to_hex(data),
                  "chainId": 80001 if self.network=="mumbai" else 137}
            signed = w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
            logger.info("[Polygon] tx=%s", tx_hash.hex())
            return tx_hash.hex()
        except Exception as e:
            logger.warning("[Polygon] Failed: %s — mock", e)
            return self._mock_tx(commitment)

    def mint_badge(self, recipient_addr: str, token_uri: str) -> Tuple[str, int]:
        w3 = self._connect()
        if not w3 or not w3.is_connected() or not self.private_key or not self.contract_address:
            token_id = abs(hash(recipient_addr + token_uri)) % 100_000
            return self._mock_tx(recipient_addr), token_id
        try:
            from web3.middleware import geth_poa_middleware  # noqa: F401 — already imported at module level
            BADGE_ABI = [{"inputs":[{"name":"recipient","type":"address"},
                                    {"name":"tokenURI","type":"string"}],
                          "name":"mint","outputs":[{"name":"tokenId","type":"uint256"}],
                          "stateMutability":"nonpayable","type":"function"}]
            acct     = w3.eth.account.from_key(self.private_key)
            contract = w3.eth.contract(address=Web3.to_checksum_address(self.contract_address),
                                       abi=BADGE_ABI)
            nonce = w3.eth.get_transaction_count(acct.address)
            tx = contract.functions.mint(
                Web3.to_checksum_address(recipient_addr), token_uri
            ).build_transaction({"from": acct.address, "nonce": nonce,
                                  "gas": 200000, "gasPrice": w3.eth.gas_price})
            signed  = w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            token_id = int(receipt["logs"][0]["data"], 16) if receipt.get("logs") else 0
            return tx_hash.hex(), token_id
        except Exception as e:
            logger.warning("[Polygon] Mint failed: %s — mock", e)
            token_id = abs(hash(recipient_addr + token_uri)) % 100_000
            return self._mock_tx(recipient_addr), token_id

    def _mock_tx(self, seed: str) -> str:
        return "0x" + hashlib.sha256(f"mock:{seed}:{time.time_ns()}".encode()).hexdigest()

    def scan_url(self, tx_hash: str) -> str:
        base = "https://mumbai.polygonscan.com" if self.network=="mumbai" else "https://polygonscan.com"
        return f"{base}/tx/{tx_hash}"


# ══════════════════════════════════════════════════════════════════════════
# Local Registry (fallback / dev)
# ══════════════════════════════════════════════════════════════════════════

class LocalProvenanceRegistry:
    """In-memory provenance store untuk dev mode / fallback."""

    def __init__(self, persist_path: Optional[str]=None):
        self._commitments: Dict[str, dict]     = {}
        self._badges:      Dict[str, ESGBadge] = {}
        self._path = persist_path

    def store_commitment(self, commitment: str, meta: dict) -> str:
        proof_id = "local-" + hashlib.sha256(
            f"{commitment}{time.time_ns()}".encode()).hexdigest()[:32]
        self._commitments[proof_id] = {
            "proof_id": proof_id, "commitment": commitment,
            "metadata": meta, "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        return proof_id

    def store_badge(self, badge: ESGBadge) -> None:
        self._badges[badge.badge_id] = badge
        self._save()

    def get_badge(self, badge_id: str) -> Optional[ESGBadge]:
        return self._badges.get(badge_id)

    def list_commitments(self) -> List[dict]:
        return list(self._commitments.values())

    def list_badges(self) -> List[ESGBadge]:
        return list(self._badges.values())

    def _save(self):
        if not self._path:
            return
        try:
            with open(self._path, "w") as f:
                json.dump({"commitments": self._commitments,
                           "badges": {bid: b.__dict__ for bid,b in self._badges.items()}},
                          f, indent=2)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
# Main Integration Engine
# ══════════════════════════════════════════════════════════════════════════

class BlockchainIntegration:
    """
    Orchestrator utama untuk semua fitur blockchain di Eco AI Data Center.

    Workflow:
      publish_audit_trail()  → ZK commit → IPFS → Polygon → ProvenanceChain
      mint_esg_badge()       → ZK commit → IPFS metadata → Polygon mint → sign

    Zero-trust: data mentah TIDAK PERNAH keluar. Hanya commitment hash on-chain.

    Parameters
    ----------
    provenance       : ProvenanceChain CATERYA (auto-create jika None)
    polygon_rpc      : RPC URL Polygon (default: Mumbai testnet)
    pinata_api_key   : Pinata API key untuk IPFS
    pinata_secret    : Pinata secret
    contract_address : Alamat smart contract badge di Polygon
    private_key      : Private key wallet Polygon
    network          : "mumbai" | "mainnet"
    signing_key      : Secret untuk HMAC signature badge
    """

    BADGE_TYPES = {
        "COS_VERIFIED":       "CATERYA Open Score — ethical AI certified",
        "ESG_REPORT":         "Full ESG sustainability report verified",
        "CARBON_NEUTRAL":     "Carbon neutrality commitment verified",
        "ISO27001_COMPLIANT": "ISO 27001 security compliance verified",
        "SRN_PPI_COMPLIANT":  "Indonesian carbon regulation (SRN PPI) compliant",
        "QUANTUM_VERIFIED":   "Quantum-validated thermal distribution",
    }

    def __init__(
        self,
        provenance:       Optional[ProvenanceChain] = None,
        polygon_rpc:      Optional[str] = None,
        pinata_api_key:   Optional[str] = None,
        pinata_secret:    Optional[str] = None,
        contract_address: Optional[str] = None,
        private_key:      Optional[str] = None,
        network:          str = "mumbai",
        signing_key:      Optional[str] = None,
        local_persist:    Optional[str] = None,
    ):
        self.provenance  = provenance or ProvenanceChain(model_id="blockchain-integration")
        self.ipfs        = IPFSPublisher(pinata_api_key=pinata_api_key, pinata_secret=pinata_secret)
        self.polygon     = PolygonPublisher(rpc_url=polygon_rpc, contract_address=contract_address,
                                            private_key=private_key, network=network)
        self.registry    = LocalProvenanceRegistry(persist_path=local_persist)
        self.signing_key = signing_key or os.getenv("BADGE_SIGNING_KEY", secrets.token_hex(32))
        self._badge_cache: Dict[str, ESGBadge] = {}

        logger.info("[Blockchain] Init | network=%s | web3=%s",
                    network, "yes" if WEB3_AVAILABLE else "mock")

    # ── publish audit trail ──────────────────────────────────────────────

    def publish_audit_trail(
        self,
        event_type:      str,
        data:            Dict[str, Any],
        actor:           str = "system",
        blinding_factor: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Publish audit trail dengan ZK commitment.

        Hanya commitment hash yang on-chain — data sensitif tetap private.
        Kembalikan blinding_factor agar caller bisa verifikasi di kemudian hari.
        """
        zk         = ZKCommitment.commit(data, blinding_factor=blinding_factor)
        commitment = zk["commitment"]

        public_record = {
            "event_type":  event_type,
            "actor_hash":  hashlib.sha256(actor.encode()).hexdigest()[:16],
            "commitment":  commitment,
            "merkle_root": zk["merkle_root"],
            "data_hash":   zk["data_hash"],
            "timestamp":   zk["timestamp"],
            "framework":   "CATERYA-v2.0",
        }

        ipfs_cid = self.ipfs.publish(public_record, name=f"caterya-{event_type}")
        tx_hash  = self.polygon.publish_commitment(
            commitment=commitment,
            metadata_uri=self.ipfs.gateway_url(ipfs_cid),
            event_type=event_type,
        )
        proof_id = self.registry.store_commitment(commitment, public_record)

        self.provenance.record(f"blockchain.{event_type}", {
            "commitment":  commitment[:16] + "...",
            "merkle_root": zk["merkle_root"][:16] + "...",
            "ipfs_cid":    ipfs_cid,
            "tx_hash":     tx_hash[:18] + "...",
            "actor":       actor,
            "network":     self.polygon.network,
        })

        return {
            "commitment":      commitment,
            "blinding_factor": zk["blinding_factor"],
            "merkle_root":     zk["merkle_root"],
            "data_hash":       zk["data_hash"],
            "ipfs_cid":        ipfs_cid,
            "ipfs_url":        self.ipfs.gateway_url(ipfs_cid),
            "tx_hash":         tx_hash,
            "polygon_scan":    self.polygon.scan_url(tx_hash),
            "proof_id":        proof_id,
            "event_type":      event_type,
            "timestamp":       zk["timestamp"],
        }

    # ── mint ESG badge ───────────────────────────────────────────────────

    def mint_esg_badge(
        self,
        recipient:          str,
        badge_type:         str,
        cos_score:          float,
        pue:                float,
        compliance_status:  str,
        simulation_data:    Optional[Dict[str, Any]] = None,
        recipient_address:  str = "0x0000000000000000000000000000000000000000",
    ) -> ESGBadge:
        """
        Mint CATERYA-Verified ESG Badge.
        ZK-commit simulation_data → publish ke IPFS → mint di Polygon → sign.
        """
        badge_id   = f"badge-{uuid.uuid4().hex[:16]}"
        sim_data   = simulation_data or {"cos_score": cos_score, "pue": pue}
        zk         = ZKCommitment.commit(sim_data)
        commitment = zk["commitment"]

        badge = ESGBadge(
            badge_id=badge_id, recipient=recipient, badge_type=badge_type,
            cos_score=cos_score, pue_score=pue, compliance_status=compliance_status,
            commitment_hash=commitment, merkle_root=zk["merkle_root"],
        )

        metadata = badge.to_metadata_dict()
        metadata["zk_proof"] = {"data_hash": zk["data_hash"],
                                "merkle_root": zk["merkle_root"],
                                "verify_at": "https://cateryatech.com/verify"}
        ipfs_cid      = self.ipfs.publish(metadata, name=f"badge-{badge_id}")
        badge.ipfs_cid     = ipfs_cid
        badge.metadata_uri = f"ipfs://{ipfs_cid}"

        tx_hash, token_id = self.polygon.mint_badge(
            recipient_address, self.ipfs.gateway_url(ipfs_cid))
        badge.tx_hash  = tx_hash
        badge.token_id = str(token_id)

        payload = f"{badge_id}:{commitment}:{badge.issued_at}"
        badge.signature = _hmac.new(
            self.signing_key.encode(), payload.encode(), hashlib.sha256).hexdigest()

        self.registry.store_badge(badge)
        self._badge_cache[badge_id] = badge

        self.provenance.record("blockchain.badge_minted", {
            "badge_id": badge_id, "badge_type": badge_type,
            "recipient": recipient, "cos_score": cos_score,
            "token_id": str(token_id), "ipfs_cid": ipfs_cid,
            "tx_hash": tx_hash[:18] + "...",
        })

        logger.info("[Blockchain] Badge minted | %s | cos=%.4f | token=%s",
                    badge_type, cos_score, token_id)
        return badge

    # ── verify ──────────────────────────────────────────────────────────

    def verify_badge(self, badge_id: str) -> Dict[str, Any]:
        badge = self._badge_cache.get(badge_id) or self.registry.get_badge(badge_id)
        if not badge:
            return {"valid": False, "error": f"Badge {badge_id} not found."}
        sig_ok = badge.verify_signature(self.signing_key)
        return {
            "valid":         sig_ok,
            "badge_id":      badge_id,
            "badge_type":    badge.badge_type,
            "recipient":     badge.recipient,
            "cos_score":     badge.cos_score,
            "pue":           badge.pue_score,
            "issued_at":     badge.issued_at,
            "signature_ok":  sig_ok,
            "tx_hash":       badge.tx_hash,
            "polygon_scan":  self.polygon.scan_url(badge.tx_hash or ""),
            "ipfs_cid":      badge.ipfs_cid,
            "ipfs_url":      self.ipfs.gateway_url(badge.ipfs_cid or ""),
            "has_blockchain": bool(badge.tx_hash),
            "has_ipfs":       bool(badge.ipfs_cid),
            "commitment":    badge.commitment_hash[:16] + "...",
            "verifiable_credential": badge.as_verifiable_credential(),
        }

    def verify_commitment(self, data: Dict[str, Any],
                          commitment: str, blinding_factor: str) -> bool:
        return ZKCommitment.verify(data, commitment, blinding_factor)

    # ── convenience wrappers ─────────────────────────────────────────────

    def publish_pue_simulation(self, pue: float, wue: float,
                                carbon_kg: float, cos_score: float,
                                actor: str = "optimizer",
                                extra: Optional[Dict] = None) -> Dict[str, str]:
        data = {"pue": round(pue,4), "wue": round(wue,4),
                "carbon_kg": round(carbon_kg,2), "cos_score": round(cos_score,4)}
        if extra:
            data.update(extra)
        return self.publish_audit_trail("pue_optimisation", data, actor)

    def publish_compliance_scan(self, framework: str, score: float,
                                 status: str, actor: str = "compliance_officer") -> Dict[str, str]:
        return self.publish_audit_trail(
            f"compliance_scan.{framework}",
            {"framework": framework, "score": score, "status": status}, actor)

    def summary(self) -> Dict[str, Any]:
        commitments = self.registry.list_commitments()
        badges      = self.registry.list_badges()
        return {
            "total_commitments":   len(commitments),
            "total_badges":        len(badges),
            "provenance_events":   len(self.provenance._chain),
            "network":             self.polygon.network,
            "web3_available":      WEB3_AVAILABLE,
            "badge_types":         list(self.BADGE_TYPES.keys()),
            "recent_commitments":  commitments[-3:],
        }
