#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터베이스 스키마 확인 스크립트
현재 데이터베이스 구조를 확인하고 새로운 파이프라인과의 호환성을 검사합니다.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import sys
from pathlib import Path

# 데이터베이스 설정
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'kkokkaot_closet',
    'user': 'postgres',
    'password': '000000'
}

class DatabaseSchemaChecker:
    """데이터베이스 스키마 확인 클래스"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
    
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            print("✅ 데이터베이스 연결 성공")
            return True
        except Exception as e:
            print(f"❌ 데이터베이스 연결 실패: {e}")
            return False
    
    def disconnect(self):
        """데이터베이스 연결 해제"""
        if self.conn:
            self.conn.close()
            print("✅ 데이터베이스 연결 해제")
    
    def get_table_info(self, table_name):
        """테이블 정보 조회"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns 
                    WHERE table_name = %s
                    ORDER BY ordinal_position
                """, (table_name,))
                return cur.fetchall()
        except Exception as e:
            print(f"❌ 테이블 정보 조회 실패: {e}")
            return []
    
    def get_all_tables(self):
        """모든 테이블 목록 조회"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"❌ 테이블 목록 조회 실패: {e}")
            return []
    
    def check_wardrobe_items_table(self):
        """wardrobe_items 테이블 확인"""
        print("\n📋 wardrobe_items 테이블 확인")
        print("-" * 50)
        
        columns = self.get_table_info('wardrobe_items')
        if not columns:
            print("❌ wardrobe_items 테이블이 존재하지 않습니다!")
            return False
        
        required_columns = {
            'style': False,
            'style_confidence': False
        }
        
        print("현재 컬럼들:")
        for col in columns:
            print(f"  - {col['column_name']}: {col['data_type']} {'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'}")
            
            if col['column_name'] in required_columns:
                required_columns[col['column_name']] = True
        
        print("\n필수 컬럼 확인:")
        all_present = True
        for col_name, is_present in required_columns.items():
            status = "✅" if is_present else "❌"
            print(f"  {status} {col_name}")
            if not is_present:
                all_present = False
        
        return all_present
    
    def check_attribute_tables(self):
        """속성 테이블들 확인"""
        print("\n📋 속성 테이블들 확인")
        print("-" * 50)
        
        tables = ['top_attributes', 'bottom_attributes', 'outer_attributes', 'dress_attributes']
        required_columns = ['material', 'length', 'sleeve_length', 'neckline', 'print_pattern']
        
        all_tables_ok = True
        
        for table_name in tables:
            print(f"\n🔍 {table_name} 테이블:")
            columns = self.get_table_info(table_name)
            
            if not columns:
                print(f"  ❌ {table_name} 테이블이 존재하지 않습니다!")
                all_tables_ok = False
                continue
            
            print("  현재 컬럼들:")
            present_columns = []
            for col in columns:
                print(f"    - {col['column_name']}: {col['data_type']}")
                present_columns.append(col['column_name'])
            
            print("  필수 컬럼 확인:")
            table_ok = True
            for col_name in required_columns:
                is_present = col_name in present_columns
                status = "✅" if is_present else "❌"
                print(f"    {status} {col_name}")
                if not is_present:
                    table_ok = False
            
            if not table_ok:
                all_tables_ok = False
        
        return all_tables_ok
    
    def check_style_analysis_table(self):
        """style_analysis 테이블 확인"""
        print("\n📋 style_analysis 테이블 확인")
        print("-" * 50)
        
        columns = self.get_table_info('style_analysis')
        if not columns:
            print("❌ style_analysis 테이블이 존재하지 않습니다!")
            return False
        
        print("현재 컬럼들:")
        for col in columns:
            print(f"  - {col['column_name']}: {col['data_type']} {'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'}")
        
        return True
    
    def check_indexes(self):
        """인덱스 확인"""
        print("\n📋 인덱스 확인")
        print("-" * 50)
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        indexname,
                        tablename,
                        indexdef
                    FROM pg_indexes 
                    WHERE schemaname = 'public'
                    AND indexname LIKE 'idx_%'
                    ORDER BY tablename, indexname
                """)
                indexes = cur.fetchall()
                
                if indexes:
                    print("현재 인덱스들:")
                    for idx in indexes:
                        print(f"  - {idx[0]} on {idx[1]}")
                else:
                    print("❌ 인덱스가 없습니다!")
                    return False
                
                return True
                
        except Exception as e:
            print(f"❌ 인덱스 확인 실패: {e}")
            return False
    
    def check_views(self):
        """뷰 확인"""
        print("\n📋 뷰 확인")
        print("-" * 50)
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT viewname 
                    FROM pg_views 
                    WHERE schemaname = 'public'
                    ORDER BY viewname
                """)
                views = cur.fetchall()
                
                if views:
                    print("현재 뷰들:")
                    for view in views:
                        print(f"  - {view[0]}")
                    
                    if 'wardrobe_with_style' in [v[0] for v in views]:
                        print("✅ wardrobe_with_style 뷰가 존재합니다!")
                        return True
                    else:
                        print("❌ wardrobe_with_style 뷰가 없습니다!")
                        return False
                else:
                    print("❌ 뷰가 없습니다!")
                    return False
                    
        except Exception as e:
            print(f"❌ 뷰 확인 실패: {e}")
            return False
    
    def get_table_counts(self):
        """테이블별 데이터 개수 확인"""
        print("\n📊 테이블별 데이터 개수")
        print("-" * 50)
        
        tables = ['wardrobe_items', 'top_attributes', 'bottom_attributes', 
                 'outer_attributes', 'dress_attributes', 'style_analysis']
        
        try:
            with self.conn.cursor() as cur:
                for table_name in tables:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cur.fetchone()[0]
                        print(f"  - {table_name}: {count:,}개")
                    except Exception as e:
                        print(f"  - {table_name}: 테이블이 존재하지 않음")
                        
        except Exception as e:
            print(f"❌ 데이터 개수 확인 실패: {e}")
    
    def run_check(self):
        """전체 스키마 확인 실행"""
        print("🔍 데이터베이스 스키마 확인 시작!")
        print("=" * 60)
        
        if not self.connect():
            return False
        
        try:
            # 기본 테이블 목록 확인
            print("\n📋 데이터베이스 테이블 목록")
            print("-" * 50)
            tables = self.get_all_tables()
            if tables:
                for table in tables:
                    print(f"  - {table}")
            else:
                print("❌ 테이블이 없습니다!")
                return False
            
            # 각 테이블별 상세 확인
            results = {
                'wardrobe_items': self.check_wardrobe_items_table(),
                'attribute_tables': self.check_attribute_tables(),
                'style_analysis': self.check_style_analysis_table(),
                'indexes': self.check_indexes(),
                'views': self.check_views()
            }
            
            # 데이터 개수 확인
            self.get_table_counts()
            
            # 결과 요약
            print("\n" + "=" * 60)
            print("📊 스키마 호환성 검사 결과")
            print("=" * 60)
            
            all_ok = True
            for check_name, is_ok in results.items():
                status = "✅ 통과" if is_ok else "❌ 실패"
                print(f"  {status}: {check_name}")
                if not is_ok:
                    all_ok = False
            
            if all_ok:
                print("\n🎉 모든 검사를 통과했습니다!")
                print("새로운 파이프라인과 완전히 호환됩니다.")
            else:
                print("\n⚠️ 일부 검사에서 실패했습니다.")
                print("database_migration.py를 실행하여 스키마를 업데이트하세요.")
            
            return all_ok
            
        except Exception as e:
            print(f"\n❌ 스키마 확인 중 오류 발생: {e}")
            return False
        
        finally:
            self.disconnect()

def main():
    """메인 실행 함수"""
    print("🎯 PostgreSQL 데이터베이스 스키마 확인 도구")
    print("새로운 패션 파이프라인과의 호환성 검사")
    print("=" * 60)
    
    # 데이터베이스 설정 확인
    print(f"📊 데이터베이스 설정:")
    print(f"  - 호스트: {DB_CONFIG['host']}")
    print(f"  - 포트: {DB_CONFIG['port']}")
    print(f"  - 데이터베이스: {DB_CONFIG['database']}")
    print(f"  - 사용자: {DB_CONFIG['user']}")
    
    # 스키마 확인 실행
    checker = DatabaseSchemaChecker(DB_CONFIG)
    success = checker.run_check()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
