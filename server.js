import express from "express";
import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
// 設定較大的傳輸限制以處理圖片
app.use(express.json({ limit: "20mb" }));
app.use(cors());

// 初始化雙 AI 客戶端
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// 選擇 Gemini 模型 (Flash 速度快且便宜，Pro 判斷力更強)
const geminiModel = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });

app.post("/api/analyze-dual", async (req, res) => {
  try {
    const { image } = req.body; // 預期格式: "data:image/jpeg;base64,..."
    if (!image) return res.status(400).json({ error: "No image provided" });

    // 擷取 Base64 純資料部分 (移除 data:image/jpeg;base64, 前綴)
    const base64Data = image.split(",")[1];
    const mimeType = image.split(";")[0].split(":")[1];

    console.log("1. 正在請求 ChatGPT (GPT-4o) 進行初步分析...");
    
    // --- 第一階段：ChatGPT 初步分析 ---
    const gptResponse = await openai.chat.completions.create({
      model: "gpt-5",
      messages: [
        {
          role: "system",
          content: `你是一位初級蔬食分析師。請分析這張食物圖片，並以 JSON 格式回傳以下資訊：
          1. 菜名 (dishName)
          2. 是否為素食 (isVegetarian: boolean)
          3. 預估熱量 (calories)
          4. 預估重量 (estimatedWeight)
          5. 食材列表 (ingredients: [{name, amount}])
          6. 簡易做法 (recipeSteps: array of strings)
          
          請直接回傳 JSON，不要 markdown 格式。`
        },
        {
          role: "user",
          content: [{ type: "image_url", image_url: { url: image } }]
        }
      ],
      response_format: { type: "json_object" } // 強制 JSON 模式
    });

    const gptResultRaw = gptResponse.choices[0].message.content;
    console.log("ChatGPT 初步分析完成，準備交給 Gemini 審核...");

    // --- 第二階段：Gemini 審核與優化 ---
    // Gemini 接收：1. 原始圖片 2. ChatGPT 的分析文字
    
    const promptForGemini = `
      你是一位頂級的蔬食總主廚與營養專家。
      這是你的助手 (ChatGPT) 對這張圖片的初步分析：
      ${gptResultRaw}

      請看著圖片，嚴格審查助手的分析：
      1. **葷素檢查**：助手是否遺漏了可能的葷食成分（如培根碎、蝦米、肉燥）？如果是，請強制將 isVegetarian 改為 false 並在菜名標註 (含葷)。
      2. **準確度修正**：如果助手把「炒空心菜」看成「炒菠菜」，請修正。
      3. **熱量與做法優化**：請提供更精準的熱量估算與更美味的做法。

      請輸出最終確認的 JSON，格式必須與助手的一致，直接輸出純 JSON 文字：
      {
        "dishName": "...",
        "isVegetarian": true/false,
        "calories": 123,
        "estimatedWeight": "...",
        "ingredients": [...],
        "recipeSteps": [...]
      }
    `;

    const geminiResult = await geminiModel.generateContent([
      promptForGemini,
      {
        inlineData: {
          data: base64Data,
          mimeType: mimeType
        }
      }
    ]);

    // 處理 Gemini 回傳 (有時會有 markdown ```json 包裹，需要清理)
    let finalContent = geminiResult.response.text();
    finalContent = finalContent.replace(/```json/g, "").replace(/```/g, "").trim();
    
    const finalJson = JSON.parse(finalContent);
    console.log("2. Gemini 審核完成，回傳最終結果。");

    // 回傳給前端
    res.json(finalJson);

  } catch (error) {
    console.error("雙 AI 分析失敗:", error);
    res.status(500).json({ error: "AI 思考過程中發生錯誤", details: error.message });
  }
});
// ==========================================
// 修正版：純文字回推 API (文字 -> GPT -> Gemini)
// ==========================================
app.post("/api/analyze-text-dual", async (req, res) => {
  try {
    const { dishName } = req.body;
    if (!dishName) return res.status(400).json({ error: "請輸入菜名" });

    console.log(`1. [文字模式] 正在請求 ChatGPT 分析: ${dishName}...`);

    // --- 第一階段：ChatGPT 初步分析 ---
    const gptResponse = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `你是一位專業蔬食營養師。請分析使用者輸入的菜名，並回傳 JSON：
          1. 菜名 (dishName): 使用者輸入的名稱
          2. 是否為素食 (isVegetarian: boolean)
          3. 預估熱量 (calories): 純數字 (kcal)
          4. 預估重量 (estimatedWeight): 純數字 (g)
          5. 食材列表 (ingredients: [{name, amount}])
          6. 簡易做法 (recipeSteps: array of strings)
          
          
          請直接回傳 JSON，不要 markdown。`
        },
        { role: "user", content: dishName }
      ],
      response_format: { type: "json_object" }
    });

    const gptResultRaw = gptResponse.choices[0].message.content;
    console.log("ChatGPT 初步分析完成，準備交給 Gemini 審核...");

    // --- 第二階段：Gemini 審核 ---
    const promptForGemini = `
      你是一位頂級蔬食主廚。
      這是助手對菜餚「${dishName}」的分析：
      ${gptResultRaw}

      請嚴格審查：
      1. **葷素檢查**：確認這道菜傳統上是否含葷？如果是，請標註 (含葷)。
      2. **合理性**：熱量與食材量是否合理？
      3. **優化**：提供更道地的做法。
      4. **篩選**：不要有蔥、蒜、韭、薤、興渠(洋蔥),如果有的話改成薑,如果這道料理不適合放薑,就不要

      請輸出最終 JSON，格式與助手一致：
      {
        "dishName": "...",
        "isVegetarian": true/false,
        "calories": 123,
        "estimatedWeight": "...",
        "ingredients": [...],
        "recipeSteps": [...]
      }
    `;

    const geminiResult = await geminiModel.generateContent(promptForGemini);
    const text = geminiResult.response.text();

    // === 🛠️ 關鍵修正：強力清洗 JSON ===
    // 透過尋找第一個 '{' 和最後一個 '}' 來擷取純 JSON，忽略前後的廢話
    const startIndex = text.indexOf('{');
    const endIndex = text.lastIndexOf('}');

    if (startIndex !== -1 && endIndex !== -1) {
        const jsonStr = text.substring(startIndex, endIndex + 1);
        const finalJson = JSON.parse(jsonStr);
        console.log("2. Gemini (文字模式) 審核完成。");
        res.json(finalJson);
    } else {
        throw new Error("AI 回傳的資料格式無法解析，請重試");
    }
    // =================================

  } catch (error) {
    console.error("文字分析失敗:", error);
    // 這裡是為了防止前端一直轉圈圈，如果失敗回傳一個預設錯誤
    res.status(500).json({ error: "AI 思考過程中發生錯誤，請再試一次" });
  }
});
const PORT = process.env.PORT || 3000;
app.listen(PORT, "0.0.0.0", () => console.log(`Dual-AI Server running on port ${PORT}`));
