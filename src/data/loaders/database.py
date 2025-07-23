"""Database loader for FTIR and HIPS data"""

import pandas as pd
import sqlite3
from typing import Optional, Dict, Any
from pathlib import Path
from core.base import BaseLoader
from core.exceptions import DataValidationError

class FTIRHIPSLoader(BaseLoader):
    """
    Database loader for FTIR and HIPS data from SQLite database
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the loader
        
        Parameters:
        -----------
        db_path : str
            Path to SQLite database
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise DataValidationError(f"Database not found: {db_path}")
    
    def load(self, site_code: str) -> pd.DataFrame:
        """
        Load FTIR and HIPS data for a specific site
        
        Parameters:
        -----------
        site_code : str
            Site code (e.g., 'ETAD')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with FTIR and HIPS measurements
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if site exists
            site_query = "SELECT DISTINCT site_code FROM filters WHERE site_code = ?"
            site_exists = pd.read_sql_query(site_query, conn, params=(site_code,))
            
            if len(site_exists) == 0:
                available_sites = self.get_available_sites()
                conn.close()
                raise DataValidationError(
                    f"Site '{site_code}' not found. Available sites: {available_sites}"
                )
            
            # Main query
            query = """
            SELECT 
                f.filter_id, f.sample_date, f.site_code, m.volume_m3,
                m.ec_ftir, m.ec_ftir_mdl, m.oc_ftir, m.oc_ftir_mdl,
                m.fabs, m.fabs_mdl, m.fabs_uncertainty, m.ftir_batch_id
            FROM filters f
            JOIN ftir_measurements m ON f.filter_id = m.filter_id
            WHERE f.site_code = ?
            ORDER BY f.sample_date
            """
            
            df = pd.read_sql_query(query, conn, params=(site_code,))
            conn.close()
            
            # Convert date column
            df['sample_date'] = pd.to_datetime(df['sample_date'])
            
            return df
            
        except sqlite3.Error as e:
            raise DataValidationError(f"Database error: {e}")
        except Exception as e:
            raise DataValidationError(f"Error loading data: {e}")
    
    def get_available_sites(self) -> list:
        """
        Get list of available sites in the database
        
        Returns:
        --------
        list
            List of available site codes
        """
        try:
            conn = sqlite3.connect(self.db_path)
            sites_df = pd.read_sql_query("SELECT DISTINCT site_code FROM filters", conn)
            conn.close()
            return sites_df['site_code'].tolist()
        except Exception:
            return []
    
    def get_data_summary(self, site_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of available data
        
        Parameters:
        -----------
        site_code : str, optional
            Site code to summarize (if None, summarize all sites)
            
        Returns:
        --------
        Dict[str, Any]
            Summary statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            if site_code:
                query = """
                SELECT 
                    COUNT(*) as total_samples,
                    MIN(sample_date) as earliest_date,
                    MAX(sample_date) as latest_date,
                    COUNT(CASE WHEN ec_ftir IS NOT NULL THEN 1 END) as ec_samples,
                    COUNT(CASE WHEN oc_ftir IS NOT NULL THEN 1 END) as oc_samples,
                    COUNT(CASE WHEN fabs IS NOT NULL THEN 1 END) as fabs_samples
                FROM filters f
                JOIN ftir_measurements m ON f.filter_id = m.filter_id
                WHERE f.site_code = ?
                """
                summary = pd.read_sql_query(query, conn, params=(site_code,))
            else:
                query = """
                SELECT 
                    f.site_code,
                    COUNT(*) as total_samples,
                    MIN(sample_date) as earliest_date,
                    MAX(sample_date) as latest_date,
                    COUNT(CASE WHEN ec_ftir IS NOT NULL THEN 1 END) as ec_samples,
                    COUNT(CASE WHEN oc_ftir IS NOT NULL THEN 1 END) as oc_samples,
                    COUNT(CASE WHEN fabs IS NOT NULL THEN 1 END) as fabs_samples
                FROM filters f
                JOIN ftir_measurements m ON f.filter_id = m.filter_id
                GROUP BY f.site_code
                """
                summary = pd.read_sql_query(query, conn)
            
            conn.close()
            return summary.to_dict('records')
            
        except Exception as e:
            raise DataValidationError(f"Error getting data summary: {e}")
