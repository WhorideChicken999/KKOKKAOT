import psycopg2
from psycopg2.extras import Json

# DB 연결
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
        print("Fixing item_id 481")
        print("="*80 + "\n")
        
        # 1. top_attributes 데이터 확인
        cur.execute("""
            SELECT item_id, category, color, fit, materials,
                   category_confidence, color_confidence, fit_confidence
            FROM top_attributes
            WHERE item_id = 481
        """)
        
        top_data = cur.fetchone()
        if top_data:
            print("Step 1: Found data in top_attributes")
            print(f"  - category: {top_data[1]}")
            print(f"  - color: {top_data[2]}")
            print(f"  - fit: {top_data[3]}")
            print(f"  - materials: {top_data[4]}")
            
            # 2. outer_attributes로 복사 (materials를 Json으로 변환)
            print("\nStep 2: Copying to outer_attributes...")
            cur.execute("""
                INSERT INTO outer_attributes (
                    item_id, category, color, fit, materials,
                    category_confidence, color_confidence, fit_confidence
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                top_data[0],  # item_id
                top_data[1],  # category
                top_data[2],  # color
                top_data[3],  # fit
                Json(top_data[4]) if top_data[4] else Json([]),  # materials
                top_data[5],  # category_confidence
                top_data[6],  # color_confidence
                top_data[7]   # fit_confidence
            ))
            
            print("  => SUCCESS: Data copied to outer_attributes")
            
            # 3. top_attributes 삭제
            print("\nStep 3: Deleting from top_attributes...")
            cur.execute("DELETE FROM top_attributes WHERE item_id = 481")
            print("  => SUCCESS: Deleted from top_attributes")
            
            # 4. wardrobe_items 수정 (has_top=False, has_outer=True)
            print("\nStep 4: Updating wardrobe_items...")
            cur.execute("""
                UPDATE wardrobe_items 
                SET has_top = FALSE, has_outer = TRUE
                WHERE item_id = 481
            """)
            print("  => SUCCESS: Updated wardrobe_items (has_top=FALSE, has_outer=TRUE)")
            
            # 커밋
            conn.commit()
            print("\n" + "="*80)
            print("ALL DONE! item_id 481 fixed successfully!")
            print("="*80 + "\n")
            
        else:
            print("ERROR: No data found in top_attributes for item_id 481")
            
        # 5. 최종 확인
        print("\n" + "-"*80)
        print("Verification:")
        print("-"*80)
        
        cur.execute("""
            SELECT has_outer, has_top, has_bottom, has_dress
            FROM wardrobe_items 
            WHERE item_id = 481
        """)
        flags = cur.fetchone()
        print(f"wardrobe_items: has_outer={flags[0]}, has_top={flags[1]}, has_bottom={flags[2]}, has_dress={flags[3]}")
        
        cur.execute("""
            SELECT category, color, fit, materials
            FROM outer_attributes 
            WHERE item_id = 481
        """)
        outer = cur.fetchone()
        if outer:
            print(f"outer_attributes: category={outer[0]}, color={outer[1]}, fit={outer[2]}, materials={outer[3]}")
        else:
            print("outer_attributes: NOT FOUND")
        
        cur.execute("""
            SELECT category, color, fit
            FROM top_attributes 
            WHERE item_id = 481
        """)
        top = cur.fetchone()
        if top:
            print(f"top_attributes: STILL EXISTS (ERROR!)")
        else:
            print("top_attributes: DELETED (CORRECT)")
        
        print("-"*80 + "\n")
        
finally:
    conn.close()

