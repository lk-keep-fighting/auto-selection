# 使用示例

## 示例 1：使用默认配置（SellerSprite）

```bash
# 直接运行，使用默认的 config.json
python3 browser_automation.py
```

**工作流程：**
1. 浏览器打开并访问 SellerSprite 产品研究页面
2. 等待你手动登录（如果需要）
3. 按 Enter 继续
4. 自动提取所有商品图片
5. 使用 YOLO 检测敏感特征
6. 保存结果到 `browser_analysis_results.json`

## 示例 2：使用自定义配置

假设你要分析其他电商网站：

### 步骤 1：创建配置文件

```bash
cp config.template.json my_site_config.json
```

### 步骤 2：修改配置

编辑 `my_site_config.json`：

```json
{
  "targetUrl": "https://example.com/products",
  "selectors": {
    "imageList": ".product-grid",
    "cardItem": ".product-item",
    "imageItem": "img.product-thumbnail",
    "asinSelector": ".product-sku",
    "titleSelector": "h3.product-name"
  }
}
```

### 步骤 3：运行

```bash
python3 browser_automation.py my_site_config.json
```

## 示例 3：仅获取图片不分析

修改 `config.json`：

```json
{
  "output": {
    "save_images": true
  }
}
```

然后在 `browser_automation.py` 中注释掉分析部分：

```python
# results = self.analyze_images(images_info)
```

## 示例 4：使用传统检测器（不用 YOLO）

修改配置：

```json
{
  "detection": {
    "use_yolo": false,
    "confidence_threshold": 0.5
  }
}
```

这样会使用基于 OpenCV 的传统检测方法。

## 示例 5：启用 Logo 检测

```json
{
  "detection": {
    "use_yolo": true,
    "enable_logo_detection": true,
    "logo_model_path": "./models/logo_detection.pt"
  }
}
```

## 示例 6：无头模式运行（后台）

```json
{
  "browser": {
    "headless": true
  }
}
```

注意：无头模式下无法手动登录，需要预先保存登录状态。

## 示例 7：程序化使用

```python
from browser_automation import BrowserAutomation

# 创建实例
automation = BrowserAutomation("config.json")

# 初始化浏览器
automation.init_browser()

# 打开网址
automation.open_url()

# 等待用户确认
automation.wait_for_user_confirmation()

# 保存登录状态
automation.save_login_state()

# 获取图片列表
images = automation.get_image_list()

# 分析图片
results = automation.analyze_images(images)

# 保存结果
automation.save_results(results)

# 关闭浏览器
automation.close()
```

## 示例 8：多页面处理

修改代码以支持翻页：

```python
def run_multiple_pages(self, num_pages=5):
    """处理多个页面"""
    all_images = []
    
    for page in range(1, num_pages + 1):
        print(f"\n处理第 {page} 页...")
        
        # 获取当前页图片
        images = self.get_image_list()
        all_images.extend(images)
        
        # 点击下一页
        try:
            next_button = self.driver.find_element(By.CSS_SELECTOR, ".next-page")
            next_button.click()
            time.sleep(2)
        except:
            print("没有更多页面")
            break
    
    return all_images
```

## 示例 9：过滤特定商品

在获取图片列表后进行过滤：

```python
# 只分析价格在某个范围内的商品
filtered_images = [
    img for img in images_info 
    if extract_price(img) >= 60 and extract_price(img) <= 99
]

results = self.analyze_images(filtered_images)
```

## 示例 10：导出为 CSV

```python
import csv

def export_to_csv(results, filename="results.csv"):
    """导出结果为 CSV"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ASIN', 'Title', 'Status', 'Features'])
        
        for item in results['detected']:
            features = ', '.join([f['feature'] for f in item.get('features', [])])
            writer.writerow([
                item.get('asin'),
                item.get('title'),
                'Detected',
                features
            ])
        
        for item in results['passed']:
            writer.writerow([
                item.get('asin'),
                item.get('title'),
                'Passed',
                ''
            ])
```

## 调试技巧

### 1. 查看浏览器日志

```python
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

caps = DesiredCapabilities.CHROME.copy()
caps['goog:loggingPrefs'] = {'browser': 'ALL'}
driver = webdriver.Chrome(desired_capabilities=caps)
```

### 2. 截图调试

```python
# 在关键步骤截图
self.driver.save_screenshot('debug_screenshot.png')
```

### 3. 暂停调试

```python
# 在需要暂停的地方
import pdb; pdb.set_trace()
```

### 4. 打印元素信息

```python
elements = self.driver.find_elements(By.CSS_SELECTOR, ".product-card")
print(f"找到 {len(elements)} 个元素")
for i, elem in enumerate(elements[:3]):
    print(f"元素 {i}: {elem.get_attribute('outerHTML')[:200]}")
```

## 常见问题解决

### Q: 选择器不工作？

**A:** 使用浏览器开发者工具验证：

```python
# 测试选择器
try:
    elements = self.driver.find_elements(By.CSS_SELECTOR, "your-selector")
    print(f"找到 {len(elements)} 个元素")
except Exception as e:
    print(f"选择器错误: {e}")
```

### Q: 元素还没加载就查找？

**A:** 添加显式等待：

```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

wait = WebDriverWait(self.driver, 10)
element = wait.until(
    EC.presence_of_element_located((By.CSS_SELECTOR, ".product-list"))
)
```

### Q: 图片是懒加载的？

**A:** 增加滚动次数和暂停时间：

```python
self._scroll_page(scroll_pause=2.0, max_scrolls=10)
```

---

更多示例和技巧，请参考：
- [浏览器自动化完整文档](BROWSER_AUTOMATION_GUIDE.md)
- [快速开始指南](QUICKSTART.md)
