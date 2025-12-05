#!/usr/bin/env python3
"""测试选择器是否正确提取页面元素"""

from playwright.sync_api import sync_playwright
import json
import re

def test_selectors():
    """测试选择器"""
    
    # 加载配置
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    selectors = config['selectors']
    
    print("🧪 测试选择器配置\n")
    print("=" * 80)
    
    with sync_playwright() as p:
        # 启动浏览器
        print("🌐 启动浏览器...")
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        
        # 打开目标网址
        target_url = config['targetUrl']
        print(f"🔗 访问: {target_url[:100]}...")
        page.goto(target_url, wait_until="domcontentloaded")
        
        # 等待用户登录
        print("\n⏸️  请手动登录（如果需要），然后按 Enter 继续测试...")
        input()
        
        print("\n🔍 开始测试选择器...\n")
        
        # 测试图片列表容器
        print(f"1️⃣ 测试图片列表容器: {selectors['imageList']}")
        container = page.query_selector(selectors['imageList'])
        if container:
            print(f"   ✅ 找到容器")
        else:
            print(f"   ❌ 未找到容器")
            browser.close()
            return
        
        # 测试商品卡片
        print(f"\n2️⃣ 测试商品卡片: {selectors['cardItem']}")
        cards = page.query_selector_all(selectors['cardItem'])
        print(f"   ✅ 找到 {len(cards)} 个商品卡片")
        
        if not cards:
            print("   ❌ 未找到任何商品卡片")
            browser.close()
            return
        
        # 测试第一个卡片的各个元素
        print(f"\n3️⃣ 测试第一个卡片的元素提取:\n")
        card = cards[0]
        
        # 测试图片
        print(f"   📷 图片选择器: {selectors['imageItem']}")
        img_element = card.query_selector(selectors['imageItem'])
        if img_element:
            style = img_element.get_attribute("style") or ""
            match = re.search(r'url\(["\']?(https?://[^"\')]+)["\']?\)', style)
            if match:
                img_url = match.group(1)
                print(f"      ✅ 图片URL: {img_url[:80]}...")
            else:
                print(f"      ⚠️  未从 style 中提取到图片URL")
                print(f"      Style: {style[:100]}")
        else:
            print(f"      ❌ 未找到图片元素")
        
        # 测试 ASIN
        print(f"\n   🔖 ASIN选择器: {selectors['asinSelector']}")
        asin_element = card.query_selector(selectors['asinSelector'])
        if asin_element:
            asin = asin_element.inner_text().strip()
            print(f"      ✅ ASIN: {asin}")
        else:
            print(f"      ❌ 未找到ASIN元素")
        
        # 测试标题
        print(f"\n   📝 标题选择器: {selectors['titleSelector']}")
        title_element = card.query_selector(selectors['titleSelector'])
        if title_element:
            title = title_element.get_attribute("title") or title_element.inner_text().strip()
            print(f"      ✅ 标题: {title[:60]}...")
        else:
            print(f"      ❌ 未找到标题元素")
        
        # 测试品牌名
        if 'brandSelector' in selectors:
            print(f"\n   🏷️  品牌选择器: {selectors['brandSelector']}")
            brand = ""
            brand_paragraphs = card.query_selector_all("p.flex-center")
            for p in brand_paragraphs:
                text = p.inner_text()
                if "品牌:" in text or "Brand:" in text.lower():
                    brand_element = p.query_selector(".over-ellipsis.text-black.sub-title")
                    if brand_element:
                        brand = brand_element.inner_text().strip()
                        break
            if brand:
                print(f"      ✅ 品牌: {brand}")
            else:
                print(f"      ❌ 未找到品牌元素")
        
        # 显示完整提取的信息
        print("\n" + "=" * 80)
        print("📊 前3个商品完整信息:\n")
        
        for idx, card in enumerate(cards[:3], 1):
            print(f"【商品 {idx}】")
            
            # 图片
            img_element = card.query_selector(selectors['imageItem'])
            if img_element:
                style = img_element.get_attribute("style") or ""
                match = re.search(r'url\(["\']?(https?://[^"\')]+)["\']?\)', style)
                img_url = match.group(1) if match else "未找到"
                print(f"  图片: {img_url}")
            
            # ASIN
            asin_element = card.query_selector(selectors['asinSelector'])
            asin = asin_element.inner_text().strip() if asin_element else "未找到"
            print(f"  ASIN: {asin}")
            
            # 标题
            title_element = card.query_selector(selectors['titleSelector'])
            if title_element:
                title = title_element.get_attribute("title") or title_element.inner_text().strip()
                print(f"  标题: {title[:60]}...")
            else:
                print(f"  标题: 未找到")
            
            # 品牌
            if 'brandSelector' in selectors:
                brand = ""
                brand_paragraphs = card.query_selector_all("p.flex-center")
                for p in brand_paragraphs:
                    text = p.inner_text()
                    if "品牌:" in text or "Brand:" in text.lower():
                        brand_element = p.query_selector(".over-ellipsis.text-black.sub-title")
                        if brand_element:
                            brand = brand_element.inner_text().strip()
                            break
                print(f"  品牌: {brand if brand else '未找到'}")
            
            print()
        
        print("=" * 80)
        print("\n✅ 测试完成！按 Enter 关闭浏览器...")
        input()
        
        browser.close()


if __name__ == "__main__":
    try:
        test_selectors()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
