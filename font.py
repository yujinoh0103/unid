import matplotlib.font_manager as fm

# Matplotlib의 폰트 관리자를 로드합니다.
# findSystemFonts=True로 시스템 폰트까지 강제로 다시 검색하게 할 수 있습니다.
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')

# 폰트 리스트가 비어있을 경우 (드물지만)
if not font_list:
    print("설치된 TTF 폰트를 찾을 수 없습니다.")
else:
    print(f"--- Matplotlib가 인식하는 폰트 (총 {len(font_list)}개) ---")
    
    # 폰트 경로로부터 '폰트 이름'을 추출하여 집합(set)에 저장 (중복 제거)
    font_names = set()
    for font_path in font_list:
        try:
            # 폰트 파일에서 실제 폰트 속성을 읽어옵니다.
            font_prop = fm.FontProperties(fname=font_path)
            font_names.add(font_prop.get_name())
        except Exception:
            # 손상된 폰트 파일 등은 무시
            pass

    # 가나다순으로 정렬하여 출력
    for i, font_name in enumerate(sorted(list(font_names))):
        print(f"{i+1:03d}: {font_name}")

print("-----------------------------------")
print("이 목록에 있는 '이름'을 rc('font', family=...)에 사용하세요.")