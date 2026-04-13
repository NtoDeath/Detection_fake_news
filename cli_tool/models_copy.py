#!/usr/bin/env python3
"""
Script de copie centralisée des modèles entraînés vers cli_tool/models/

Modèles à copier:
1. Style: RoBERTa fine-tuned + RandomForest ensemble
2. Knowledge: DistilBERT claim detector (+ DeBERTa pré-trainé runtime)
3. Fusion: Stacked RandomForest meta-learner

Utilisation:
    python models_copy.py                    # Mode interactif
    python models_copy.py --style-only        # Copie sélective
    python models_copy.py --verify-only       # Vérifier sans copier
"""

import sys
import shutil
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pickle


class ModelsCopier:
    """Orchestrate copying of all trained models to cli_tool/models/"""
    
    def __init__(self, project_root=None, verbose=True):
        """
        Initialize copier with paths
        
        Parameters
        ----------
        project_root : Path, optional
            Root of Detection_fake_news project (default: parent of cli_tool)
        verbose : bool
            Print progress messages
        """
        if project_root is None:
            # Auto-detect: cli_tool is in project root
            project_root = Path(__file__).parent.parent
        
        self.project_root = Path(project_root)
        self.cli_root = self.project_root / 'cli_tool'
        self.models_dest = self.cli_root / 'models'
        self.verbose = verbose
        
        # Tracking
        self.copied_files = []
        self.errors = []
        self.total_size = 0
    
    # ===== STYLE BRANCH =====
    def copy_style_models(self, force=False):
        """Copy RoBERTa fine-tuned + best_model.pkl"""
        print("\n🎨 STYLE BRANCH MODELS")
        print("=" * 60)
        
        try:
            # Create destination
            style_dest = self.models_dest / "style"
            style_dest.mkdir(parents=True, exist_ok=True)
            
            # 1. Copy roberta_fine_tunned/
            roberta_src = self.project_root / "style_branch" / "roberta_fine_tunned"
            roberta_dst = style_dest / "roberta_fine_tunned"
            
            if roberta_src.exists():
                self._copy_directory(roberta_src, roberta_dst, force)
                roberta_size = self._get_dir_size(roberta_src)
                print(f"   ✅ Copied: roberta_fine_tunned/ (~{roberta_size / 1024 / 1024:.1f} MB)")
            else:
                raise FileNotFoundError(f"Source not found: {roberta_src}")
            
            # 2. Copy best_model.pkl
            model_src = self.project_root / "style_branch" / "results" / "best_model.pkl"
            model_dst = style_dest / "best_model.pkl"
            
            if model_src.exists():
                self._copy_file(model_src, model_dst, force)
                print(f"   ✅ Copied: best_model.pkl (~{model_src.stat().st_size / 1024:.0f}K)")
            else:
                raise FileNotFoundError(f"Source not found: {model_src}")
            
        except Exception as e:
            error_msg = f"❌ Style copy failed: {e}"
            print(error_msg)
            self.errors.append(error_msg)
            raise
    
    # ===== KNOWLEDGE BRANCH =====
    def copy_knowledge_models(self, force=False):
        """Copy DistilBERT claim detector"""
        print("\n🧠 KNOWLEDGE BRANCH MODELS")
        print("=" * 60)
        
        try:
            # Create destination
            knowledge_dest = self.models_dest / "knowledge"
            knowledge_dest.mkdir(parents=True, exist_ok=True)
            
            # Copy my_claim_model/
            claim_src = self.project_root / "knowledge_branch" / "my_claim_model"
            claim_dst = knowledge_dest / "my_claim_model"
            
            if claim_src.exists():
                self._copy_directory(claim_src, claim_dst, force)
                claim_size = self._get_dir_size(claim_src)
                print(f"   ✅ Copied: my_claim_model/ (~{claim_size / 1024 / 1024:.1f} MB)")
            else:
                raise FileNotFoundError(f"Source not found: {claim_src}")
            
        except Exception as e:
            error_msg = f"❌ Knowledge copy failed: {e}"
            print(error_msg)
            self.errors.append(error_msg)
            raise
    
    # ===== FUSION BRANCH =====
    def copy_fusion_models(self, force=False):
        """Copy Stacked RandomForest model"""
        print("\n⭐ FUSION BRANCH MODELS")
        print("=" * 60)
        
        try:
            # Create destination
            fusion_dest = self.models_dest / "fusion"
            fusion_dest.mkdir(parents=True, exist_ok=True)
            
            # Copy stacked_rf_model.pkl
            fusion_src = self.project_root / "fusion_branch" / "results" / "stacked_rf_model.pkl"
            fusion_dst = fusion_dest / "stacked_rf_model.pkl"
            
            if fusion_src.exists():
                self._copy_file(fusion_src, fusion_dst, force)
                print(f"   ✅ Copied: stacked_rf_model.pkl (~{fusion_src.stat().st_size / 1024:.0f}K)")
            else:
                print(f"   ⚠️  Model not found: {fusion_src}")
                print(f"      Instructions: cd fusion_branch && python 07_strategy_5.py")
                self.errors.append(f"Fusion model missing: {fusion_src}")
        
        except Exception as e:
            error_msg = f"❌ Fusion copy failed: {e}"
            print(error_msg)
            self.errors.append(error_msg)
    
    # ===== VALIDATION =====
    def validate_all_models(self):
        """Test integrity of all copied models"""
        print("\n✅ VALIDATION")
        print("=" * 60)
        
        validation_errors = []
        
        # Test Style models - check file existence only (pickle load requires xgboost)
        try:
            style_pkl = self.models_dest / "style" / "best_model.pkl"
            if style_pkl.exists() and style_pkl.stat().st_size > 0:
                print("   ✅ Style model pickle exists")
            else:
                raise FileNotFoundError("Style model not found or empty")
        except Exception as e:
            validation_errors.append(f"Style model invalid: {e}")
            print(f"   ❌ Style model invalid: {e}")
        
        # Test Knowledge models - check JSON config
        try:
            claim_config = self.models_dest / "knowledge" / "my_claim_model" / "config.json"
            if claim_config.exists():
                with open(claim_config) as f:
                    json.load(f)
                print("   ✅ Knowledge config valid")
            else:
                raise FileNotFoundError("Knowledge config not found")
        except Exception as e:
            validation_errors.append(f"Knowledge model invalid: {e}")
            print(f"   ❌ Knowledge model invalid: {e}")
        
        # Test Fusion models - check file existence
        try:
            fusion_pkl = self.models_dest / "fusion" / "stacked_rf_model.pkl"
            if fusion_pkl.exists() and fusion_pkl.stat().st_size > 0:
                print("   ✅ Fusion model pickle exists")
            else:
                print("   ⚠️  Fusion model not found (optional at this stage)")
        except Exception as e:
            print(f"   ⚠️  Fusion model check skipped: {e}")
        
        return validation_errors
    
    # ===== MANIFEST =====
    def generate_manifest(self):
        """Create MANIFEST.json inventory"""
        print("\n📋 MANIFEST")
        print("=" * 60)
        
        # Calculate sizes
        style_size = 0
        if (self.models_dest / "style").exists():
            style_size = self._get_dir_size(self.models_dest / "style")
        
        knowledge_size = 0
        if (self.models_dest / "knowledge").exists():
            knowledge_size = self._get_dir_size(self.models_dest / "knowledge")
        
        fusion_size = 0
        if (self.models_dest / "fusion").exists():
            fusion_size = self._get_dir_size(self.models_dest / "fusion")
        
        manifest = {
            'timestamp': str(Path.cwd()),
            'models': {
                'style': {
                    'roberta_fine_tunned': str(self.models_dest / "style" / "roberta_fine_tunned"),
                    'best_model': str(self.models_dest / "style" / "best_model.pkl"),
                    'size_mb': round(style_size / 1024 / 1024, 1)
                },
                'knowledge': {
                    'my_claim_model': str(self.models_dest / "knowledge" / "my_claim_model"),
                    'size_mb': round(knowledge_size / 1024 / 1024, 1)
                },
                'fusion': {
                    'stacked_rf_model': str(self.models_dest / "fusion" / "stacked_rf_model.pkl"),
                    'size_mb': round(fusion_size / 1024 / 1024, 2)
                }
            },
            'total_size_mb': round((style_size + knowledge_size + fusion_size) / 1024 / 1024, 1)
        }
        
        manifest_path = self.models_dest / "MANIFEST.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"   ✅ Manifest saved: {manifest_path}")
        print(f"   📊 Total size: {manifest['total_size_mb']:.1f} MB")
        
        return manifest
    
    # ===== HELPERS =====
    def _copy_file(self, src, dst, force):
        """Copy file with overwrite option"""
        if dst.exists() and not force:
            raise FileExistsError(f"{dst} already exists. Use --force to overwrite.")
        if dst.exists() and force:
            dst.unlink()
        shutil.copy2(src, dst)
        self.total_size += src.stat().st_size
    
    def _copy_directory(self, src, dst, force):
        """Copy directory recursively"""
        if dst.exists() and force:
            shutil.rmtree(dst)
        if dst.exists():
            raise FileExistsError(f"{dst} already exists. Use --force to overwrite.")
        shutil.copytree(src, dst)
        self.total_size += self._get_dir_size(src)
    
    def _get_dir_size(self, path):
        """Get total size of directory"""
        total = 0
        for f in path.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
        return total
    
    def run(self, style=True, knowledge=True, fusion=True, force=False, verify_only=False):
        """Execute copy operations"""
        print("\n" + "=" * 70)
        print("🚀 MODELS COPIER - CLI TOOL")
        print("=" * 70)
        print(f"   Project root: {self.project_root}")
        print(f"   Destination: {self.models_dest}")
        
        if verify_only:
            print("\n🔍 VERIFY MODE (no copying)")
        
        if style and not verify_only:
            self.copy_style_models(force)
        if knowledge and not verify_only:
            self.copy_knowledge_models(force)
        if fusion and not verify_only:
            self.copy_fusion_models(force)
        
        # Validate
        validation_errors = self.validate_all_models()
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        
        if self.total_size > 0:
            print(f"   Total size copied: {self.total_size / 1024 / 1024:.1f} MB")
        
        if self.errors:
            print(f"\n   ⚠️  {len(self.errors)} issue(s):")
            for err in self.errors:
                print(f"      - {err}")
        
        if validation_errors:
            print(f"\n   ❌ {len(validation_errors)} validation error(s):")
            for err in validation_errors:
                print(f"      - {err}")
            return False
        else:
            print(f"\n   ✅ All available models validated successfully!")
        
        # Generate manifest
        manifest = self.generate_manifest()
        
        success = len(self.errors) == 0 and len(validation_errors) == 0
        
        print("\n" + "=" * 70)
        if success:
            print("✅ MODELS COPY COMPLETED SUCCESSFULLY")
        else:
            print("⚠️  MODELS COPY COMPLETED WITH WARNINGS")
        print("=" * 70 + "\n")
        
        return success


# ===== CLI INTERFACE =====
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Copy trained models to cli_tool/models/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy all models
  python models_copy.py
  
  # Copy only style models
  python models_copy.py --style-only
  
  # Verify without copying
  python models_copy.py --verify-only
  
  # Force overwrite existing
  python models_copy.py --force
        """
    )
    
    parser.add_argument('--style-only', action='store_true', help='Copy only style models')
    parser.add_argument('--knowledge-only', action='store_true', help='Copy only knowledge models')
    parser.add_argument('--fusion-only', action='store_true', help='Copy only fusion models')
    parser.add_argument('--verify-only', action='store_true', help='Verify without copying')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    
    args = parser.parse_args()
    
    copier = ModelsCopier(project_root=Path(__file__).parent.parent)
    
    # Determine what to copy
    all_selected = not any([args.style_only, args.knowledge_only, args.fusion_only])
    
    success = copier.run(
        style=args.style_only or all_selected,
        knowledge=args.knowledge_only or all_selected,
        fusion=args.fusion_only or all_selected,
        force=args.force,
        verify_only=args.verify_only
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
