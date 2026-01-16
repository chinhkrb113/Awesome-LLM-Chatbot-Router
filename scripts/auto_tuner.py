import yaml
import logging
from typing import List, Dict, Any
import shutil
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class AutoTuner:
    def __init__(self, catalog_path: str):
        self.catalog_path = catalog_path
        self.backup_dir = os.path.join(os.path.dirname(catalog_path), "backups")
        
    def tune(self, candidates_map: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Cập nhật seed phrases cho các action dựa trên candidates.
        candidates_map: {action_id: [phrase1, phrase2, ...]}
        """
        if not candidates_map:
            return {"status": "skipped", "reason": "no_candidates"}

        # 1. Backup current catalog
        self._backup_catalog()

        # 2. Load current catalog
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load catalog for tuning: {e}")
            return {"status": "error", "message": str(e)}

        actions = data.get("action_catalog", [])
        updated_count = 0
        added_phrases = 0

        # 3. Update logic
        for action in actions:
            aid = action.get("action_id")
            if aid in candidates_map:
                current_seeds = set(action.get("seed_phrases", []))
                new_candidates = candidates_map[aid]
                
                # Filter duplicates
                to_add = [p for p in new_candidates if p not in current_seeds]
                
                if to_add:
                    action["seed_phrases"] = list(current_seeds) + to_add
                    updated_count += 1
                    added_phrases += len(to_add)
                    logger.info(f"Action {aid}: Added {len(to_add)} new seed phrases.")

        # 4. Save back if changes made
        if updated_count > 0:
            try:
                with open(self.catalog_path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)
                return {
                    "status": "success", 
                    "updated_actions": updated_count, 
                    "added_phrases": added_phrases
                }
            except Exception as e:
                logger.error(f"Failed to save tuned catalog: {e}")
                # Restore backup? Maybe manual intervention is better here.
                return {"status": "error", "message": str(e)}
        
        return {"status": "skipped", "reason": "no_new_phrases"}

    def _backup_catalog(self):
        os.makedirs(self.backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"action_catalog_{timestamp}.yaml")
        shutil.copy2(self.catalog_path, backup_path)
        logger.info(f"Backed up catalog to {backup_path}")
