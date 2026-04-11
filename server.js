/*
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

// 選擇 Gemini 模型 (使用高階的 2.5 Pro)
const geminiModel = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });

// ==========================================
// API 1: 圖片回推 (圖片 -> GPT -> Gemini)
// ==========================================
app.post("/api/analyze-dual", async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) return res.status(400).json({ error: "No image provided" });

    // 擷取 Base64 純資料部分
    const base64Data = image.split(",")[1];
    const mimeType = image.split(";")[0].split(":")[1];

    console.log("1. [圖片模式] 正在請求 ChatGPT 進行初步分析...");
    
    // --- 第一階段：ChatGPT 初步分析 ---
    const gptResponse = await openai.chat.completions.create({
      model: "gpt-4o-mini", // 修正：改成 gpt-4o-mini，gpt-5 目前還不能用
      messages: [
        {
          role: "system",
          content: `你是一位專業的「無五辛蔬食」營養師。請分析這張食物圖片，並以 JSON 格式回傳以下資訊：
          1. 菜名 (dishName)
          2. 是否為素食 (isVegetarian: boolean)
          3. 預估熱量 (calories)
          4. 預估重量 (estimatedWeight)
          5. 食材列表 (ingredients: [{name, amount}])
          6. 簡易做法 (recipeSteps: array of strings)
          7. 五辛調整 (pungentAdjustment): 若發現蔥、蒜、韭、洋蔥、興渠，請優先替換成「薑」，不適合放薑則移除。若無五辛則回傳「無須調整」。
          
          請直接回傳 JSON，不要 markdown 格式。`
        },
        {
          role: "user",
          content: [{ type: "image_url", image_url: { url: image } }]
        }
      ],
      response_format: { type: "json_object" }
    });

    const gptResultRaw = gptResponse.choices[0].message.content;
    console.log("ChatGPT 初步分析完成，準備交給 Gemini 審核...");

    // --- 第二階段：Gemini 審核與優化 ---
    const promptForGemini = `
      你是一位頂級的「無五辛蔬食」總主廚。
      這是助手對這張圖片的初步分析：
      ${gptResultRaw}

      請看著圖片，嚴格審查：
      1. **葷素檢查**：確認是否含葷？(如蝦米、肉燥)，若有請標註(含葷)。
      2. **去五辛 (最高原則)**：檢查食材與步驟，絕對**不能**出現蔥、蒜、韭菜、洋蔥、興渠。
         - 如果有，請**優先改成薑**。
         - 如果這道菜不適合放薑，就**直接刪除**該食材。
      3. **合理性**：熱量與食材量是否合理？
      4. **美味優化**：提供更道地的做法。

      請輸出最終 JSON，格式與助手一致：
      {
        "dishName": "...",
        "isVegetarian": true/false,
        "calories": 123,
        "estimatedWeight": "...",
        "ingredients": [...],
        "recipeSteps": [...],
        "pungentAdjustment": "..."
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

    let text = geminiResult.response.text();
    
    // === 🛠️ 強力清洗 JSON ===
    const startIndex = text.indexOf('{');
    const endIndex = text.lastIndexOf('}');

    if (startIndex !== -1 && endIndex !== -1) {
        const jsonStr = text.substring(startIndex, endIndex + 1);
        const finalJson = JSON.parse(jsonStr);
        console.log("2. Gemini (圖片模式) 審核完成。");
        res.json(finalJson);
    } else {
        throw new Error("AI 回傳格式錯誤");
    }

  } catch (error) {
    console.error("雙 AI 圖片分析失敗:", error);
    res.status(500).json({ error: "分析失敗", details: error.message });
  }
});

// ==========================================
// API 2: 純文字回推 (文字 -> GPT -> Gemini)
// ==========================================
app.post("/api/analyze-text-dual", async (req, res) => {
  try {
    const { dishName } = req.body;
    if (!dishName) return res.status(400).json({ error: "請輸入菜名" });

    console.log(`1. [文字模式] 正在請求 ChatGPT 分析: ${dishName}...`);

    // --- 第一階段：ChatGPT 初步分析 ---
    const gptResponse = await openai.chat.completions.create({
      model: "gpt-5",
      messages: [
        {
          role: "system",
          content: `你是一位專業的「無五辛蔬食」營養師。請分析使用者輸入的菜名，並回傳 JSON：
          1. 菜名 (dishName)
          2. 是否為素食 (isVegetarian: boolean)
          3. 預估熱量 (calories): 純數字 (kcal)
          4. 預估重量 (estimatedWeight): 純數字 (g)
          5. 食材列表 (ingredients: [{name, amount}])
          6. 簡易做法 (recipeSteps: array of strings)
          7. 五辛調整 (pungentAdjustment): 若傳統做法含五辛，請優先替換成「薑」，不適合則移除。
          
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
      你是一位頂級的「無五辛蔬食」總主廚。
      這是助手對菜餚「${dishName}」的分析：
      ${gptResultRaw}

      請嚴格審查：
      1. **葷素檢查**：確認這道菜是否已調整為素食？
      2. **去五辛 (最高原則)**：檢查食材與步驟，絕對**不能**出現蔥、蒜、韭菜、洋蔥、興渠。
         - 如果有，請**優先改成薑**。
         - 如果這道菜不適合放薑，就**直接刪除**該食材。
      3. **合理性**：熱量與食材量是否合理？
      4. **美味優化**：提供更道地的無五辛素食做法。

      請輸出最終 JSON，格式與助手一致：
      {
        "dishName": "...",
        "isVegetarian": true/false,
        "calories": 123,
        "estimatedWeight": "...",
        "ingredients": [...],
        "recipeSteps": [...],
        "pungentAdjustment": "..."
      }
    `;

    const geminiResult = await geminiModel.generateContent(promptForGemini);
    const text = geminiResult.response.text();

    // === 🛠️ 強力清洗 JSON ===
    const startIndex = text.indexOf('{');
    const endIndex = text.lastIndexOf('}');

    if (startIndex !== -1 && endIndex !== -1) {
        const jsonStr = text.substring(startIndex, endIndex + 1);
        const finalJson = JSON.parse(jsonStr);
        console.log("2. Gemini (文字模式) 審核完成。");
        res.json(finalJson);
    } else {
        throw new Error("AI 回傳格式錯誤");
    }

  } catch (error) {
    console.error("文字分析失敗:", error);
    res.status(500).json({ error: "分析失敗", details: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, "0.0.0.0", () => console.log(`Dual-AI Server running on port ${PORT}`));
*/
import "./server-openai.js";
