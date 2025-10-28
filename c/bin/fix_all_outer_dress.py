import psycopg2
from psycopg2.extras import Json

conn = psycopg2.connect(
    host='localhost',
    port=5432,
    database='kkokkaot_closet',
    user='postgres',
    password='000000'
)

try:
    with conn.cursor() as cur:
        print("\n" + "="*80)
        print("FIXING ALL outer/dress misplaced data")
        print("="*80 + "\n")
        
        # ============================================================
        # PART 1: Fix OUTER items
        # ============================================================
        print("[PART 1] Fixing OUTER items...")
        print("-"*80)
        
        # has_outer=True이고 outer_attributes가 없는 아이템들
        cur.execute("""
            SELECT w.item_id, w.has_top, w.has_bottom
            FROM wardrobe_items w
            LEFT JOIN outer_attributes o ON w.item_id = o.item_id
            WHERE w.has_outer = TRUE AND o.outer_id IS NULL
            ORDER BY w.item_id
        """)
        
        outer_items = cur.fetchall()
        print(f"Found {len(outer_items)} items to fix")
        
        fixed_outer = 0
        for item in outer_items:
            item_id = item[0]
            has_top = item[1]
            has_bottom = item[2]
            
            # top_attributes에서 데이터 가져오기
            cur.execute("""
                SELECT category, color, fit, materials,
                       category_confidence, color_confidence, fit_confidence
                FROM top_attributes
                WHERE item_id = %s
            """, (item_id,))
            
            top_data = cur.fetchone()
            if top_data:
                # outer_attributes로 복사
                cur.execute("""
                    INSERT INTO outer_attributes (
                        item_id, category, color, fit, materials,
                        category_confidence, color_confidence, fit_confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    item_id,
                    top_data[0],  # category
                    top_data[1],  # color
                    top_data[2],  # fit
                    Json(top_data[3]) if top_data[3] else Json([]),  # materials
                    top_data[4],  # category_confidence
                    top_data[5],  # color_confidence
                    top_data[6]   # fit_confidence
                ))
                
                # has_top도 False여야 하는 경우 wardrobe_items 수정
                if not has_top:
                    # top_attributes 삭제
                    cur.execute("DELETE FROM top_attributes WHERE item_id = %s", (item_id,))
                
                fixed_outer += 1
                if fixed_outer % 10 == 0:
                    print(f"  Progress: {fixed_outer}/{len(outer_items)}")
        
        print(f"=> Fixed {fixed_outer} outer items")
        
        # ============================================================
        # PART 2: Fix DRESS items
        # ============================================================
        print("\n[PART 2] Fixing DRESS items...")
        print("-"*80)
        
        # has_dress=True이고 dress_attributes가 없는 아이템들
        cur.execute("""
            SELECT w.item_id, w.has_top, w.has_bottom
            FROM wardrobe_items w
            LEFT JOIN dress_attributes d ON w.item_id = d.item_id
            WHERE w.has_dress = TRUE AND d.dress_id IS NULL
            ORDER BY w.item_id
        """)
        
        dress_items = cur.fetchall()
        print(f"Found {len(dress_items)} items to fix")
        
        fixed_dress = 0
        for item in dress_items:
            item_id = item[0]
            has_top = item[1]
            has_bottom = item[2]
            
            # top_attributes에서 데이터 가져오기
            cur.execute("""
                SELECT category, color, fit, materials,
                       category_confidence, color_confidence, fit_confidence
                FROM top_attributes
                WHERE item_id = %s
            """, (item_id,))
            
            top_data = cur.fetchone()
            if top_data:
                # dress_attributes로 복사
                cur.execute("""
                    INSERT INTO dress_attributes (
                        item_id, category, color, fit, materials,
                        category_confidence, color_confidence, fit_confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    item_id,
                    top_data[0],  # category
                    top_data[1],  # color
                    top_data[2],  # fit
                    Json(top_data[3]) if top_data[3] else Json([]),  # materials
                    top_data[4],  # category_confidence
                    top_data[5],  # color_confidence
                    top_data[6]   # fit_confidence
                ))
                
                # has_top도 False여야 하는 경우 wardrobe_items 수정 및 top_attributes 삭제
                if not has_top:
                    cur.execute("DELETE FROM top_attributes WHERE item_id = %s", (item_id,))
                
                fixed_dress += 1
                if fixed_dress % 10 == 0:
                    print(f"  Progress: {fixed_dress}/{len(dress_items)}")
        
        print(f"=> Fixed {fixed_dress} dress items")
        
        # ============================================================
        # COMMIT
        # ============================================================
        print("\n" + "="*80)
        print("Committing changes...")
        conn.commit()
        print("=> SUCCESS!")
        
        # ============================================================
        # VERIFICATION
        # ============================================================
        print("\n" + "="*80)
        print("VERIFICATION:")
        print("="*80)
        
        cur.execute("SELECT COUNT(*) FROM outer_attributes")
        outer_count = cur.fetchone()[0]
        print(f"outer_attributes rows: {outer_count}")
        
        cur.execute("SELECT COUNT(*) FROM dress_attributes")
        dress_count = cur.fetchone()[0]
        print(f"dress_attributes rows: {dress_count}")
        
        # 여전히 문제가 있는 아이템 확인
        cur.execute("""
            SELECT COUNT(*)
            FROM wardrobe_items w
            LEFT JOIN outer_attributes o ON w.item_id = o.item_id
            WHERE w.has_outer = TRUE AND o.outer_id IS NULL
        """)
        still_missing_outer = cur.fetchone()[0]
        
        cur.execute("""
            SELECT COUNT(*)
            FROM wardrobe_items w
            LEFT JOIN dress_attributes d ON w.item_id = d.item_id
            WHERE w.has_dress = TRUE AND d.dress_id IS NULL
        """)
        still_missing_dress = cur.fetchone()[0]
        
        print(f"\nRemaining issues:")
        print(f"  - Missing outer_attributes: {still_missing_outer}")
        print(f"  - Missing dress_attributes: {still_missing_dress}")
        
        if still_missing_outer == 0 and still_missing_dress == 0:
            print("\nALL FIXED! No more issues!")
        
        print("="*80 + "\n")
        
finally:
    conn.close()

