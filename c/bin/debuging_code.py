import psycopg2
from psycopg2.extras import RealDictCursor

def show_all_tables():
    """모든 테이블 목록 조회"""
    
    # DB 연결
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='kkokkaot_closet',
        user='postgres',
        password='000000'
    )
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            print("\n" + "="*80)
            print("📊 데이터베이스 테이블 목록")
            print("="*80 + "\n")
            
            # 1. 모든 테이블 목록
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            
            tables = cur.fetchall()
            print(f"총 {len(tables)}개의 테이블:\n")
            
            for table in tables:
                table_name = table['table_name']
                print(f"📁 {table_name}")
                
                # 각 테이블의 컬럼 정보
                cur.execute(f"""
                    SELECT 
                        column_name, 
                        data_type, 
                        character_maximum_length,
                        is_nullable,
                        column_default
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                """)
                
                columns = cur.fetchall()
                print(f"   컬럼 수: {len(columns)}\n")
                
                for col in columns:
                    col_name = col['column_name']
                    data_type = col['data_type']
                    max_length = col['character_maximum_length']
                    nullable = col['is_nullable']
                    default = col['column_default']
                    
                    # 타입 표시
                    if max_length:
                        type_str = f"{data_type}({max_length})"
                    else:
                        type_str = data_type
                    
                    # NULL 가능 여부
                    null_str = "NULL" if nullable == "YES" else "NOT NULL"
                    
                    # 기본값
                    default_str = f" DEFAULT {default}" if default else ""
                    
                    print(f"      ├─ {col_name:<30} {type_str:<25} {null_str:<10} {default_str}")
                
                print("\n")
            
            print("="*80 + "\n")
            
            # 2. 외래키 관계
            print("\n🔗 외래키 관계:\n")
            cur.execute("""
                SELECT
                    tc.table_name, 
                    kcu.column_name, 
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name 
                FROM information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' 
                ORDER BY tc.table_name
            """)
            
            fks = cur.fetchall()
            for fk in fks:
                print(f"   {fk['table_name']}.{fk['column_name']} → {fk['foreign_table_name']}.{fk['foreign_column_name']}")
            
            print("\n" + "="*80 + "\n")
            
            # 3. 인덱스 정보
            print("\n📇 인덱스 목록:\n")
            cur.execute("""
                SELECT
                    tablename,
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """)
            
            indexes = cur.fetchall()
            current_table = None
            for idx in indexes:
                if current_table != idx['tablename']:
                    current_table = idx['tablename']
                    print(f"\n📁 {current_table}:")
                print(f"   ├─ {idx['indexname']}")
            
            print("\n" + "="*80 + "\n")
            
    finally:
        conn.close()


def show_table_data_sample(table_name: str, limit: int = 5):
    """특정 테이블의 샘플 데이터 조회"""
    
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='kkokkaot_closet',
        user='postgres',
        password='000000'
    )
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            print(f"\n📊 {table_name} 테이블 샘플 데이터 (최대 {limit}개):\n")
            
            # 데이터 수 확인
            cur.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            total_count = cur.fetchone()['count']
            print(f"   총 데이터 수: {total_count}개\n")
            
            if total_count > 0:
                # 샘플 데이터 조회
                cur.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
                rows = cur.fetchall()
                
                # 첫 번째 행의 키 가져오기
                if rows:
                    keys = rows[0].keys()
                    
                    # 테이블 형식으로 출력
                    for row in rows:
                        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        for key in keys:
                            value = row[key]
                            # JSON 데이터는 예쁘게 출력
                            if isinstance(value, (dict, list)):
                                import json
                                value = json.dumps(value, ensure_ascii=False, indent=4)
                            print(f"   {key:<30}: {value}")
                        print()
            else:
                print("   (데이터 없음)\n")
            
    finally:
        conn.close()


if __name__ == '__main__':
    # 1. 모든 테이블 구조 출력
    show_all_tables()
    
    # 2. 특정 테이블의 샘플 데이터 확인 (원하는 테이블로 변경 가능)
    print("\n" + "="*80)
    print("📋 샘플 데이터 조회")
    print("="*80)
    
    show_table_data_sample('users', limit=3)
    show_table_data_sample('wardrobe_items', limit=3)
    show_table_data_sample('top_attributes', limit=3)
    show_table_data_sample('bottom_attributes', limit=3)