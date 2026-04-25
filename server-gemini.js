import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();
app.use(express.json({ limit: "20mb" }));
app.use(cors());

const PORT = Number(process.env.PORT || 3000);
const MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";

const genAI = GEMINI_API_KEY ? new GoogleGenerativeAI(GEMINI_API_KEY) : null;

const SYSTEM_PROMPT = `
你是台灣料理與「無五辛蔬食」改作分析助手。
請根據使用者提供的菜名或菜餚照片，推測最可能的成品名稱，並輸出符合 JSON Schema 的繁體中文結果。

最高守則（違反任一條視為錯誤）：
A. recipeSteps 與 ingredients **絕對不可** 出現五辛（蔥、青蔥、洋蔥、紅蔥頭、蒜、大蒜、蒜頭、蒜末、韭菜、韭黃、薤、興渠）。
   - 需要爆香時改用薑（薑末、薑絲、薑泥）或直接省略。
   - 需要香氣時改用九層塔、香菜、芹菜、白胡椒、香菇、麻油、白芝麻油、辣椒等。
B. recipeSteps 與 ingredients **絕對不可** 出現任何葷食（豬、牛、雞、鴨、鵝、羊、魚、蝦、蟹、貝、蛤、蚵、海鮮、肉燥、絞肉、培根、火腿、香腸、臘肉、雞高湯、豬骨高湯、魚露、蝦醬、蠔油等）。
   - 若原料理常見版本含葷，請改寫成最接近原風味的蔬食版本：
     * 豬／牛／羊絞肉 → 植物絞肉、香菇丁、豆乾末
     * 雞胸／雞腿肉 → 杏鮑菇、素雞
     * 魚／蝦／海鮮 → 杏鮑菇、蒟蒻、香菇
     * 高湯 → 香菇昆布高湯或蔬菜高湯
     * 蠔油 → 純素蠔油（香菇蠔油）
   - **此時 isVegetarian 仍標記為 false**（代表「原版含葷，已自動改寫成蔬食版本」），由前端顯示「含葷食警示」。

欄位規則：
1. dishName：最可能的菜名；只有真的無法判斷時才寫「無法判定」。
2. isVegetarian：boolean。表示「這道菜的常見版本」是否本來就是素食。
   - 是素食 → true。
   - 原版含葷但已被你改寫為蔬食 → false（仍保留 false 以提示使用者原版含葷）。
3. calories：整數 kcal，估整份料理。
4. estimatedWeight：以 g 為主，可寫範圍，例如「320-380g」。
5. ingredients：4 到 10 項主要食材或調味，amount 寫實用估計值。所有項目必須是無五辛、無葷食的蔬食版本。
6. recipeSteps：1 到 8 個可實作的步驟，整份做法即可上桌。所有步驟必須完全符合無五辛、無葷食。
7. pungentAdjustment：用一句話說明你做了哪些五辛／葷食替換（例如：「以薑末取代蒜末，雞肉改為杏鮑菇」），若原本即無須調整則寫「原食譜本身即為無五辛蔬食，未做調整」。
8. 全部使用繁體中文。
9. 不要輸出 markdown、前言、後記，只回傳符合 schema 的 JSON。
10. 如果使用者已提供明確菜名，就直接根據菜名完成推測，不要回覆「未提供圖片或菜名」或要求補資料。

重要：寧可保守換掉、也不要漏掉任何五辛或葷食字眼。
`;

const MEAL_ANALYSIS_SCHEMA = {
  type: "object",
  properties: {
    dishName: { type: "string" },
    isVegetarian: { type: "boolean" },
    calories: { type: "integer" },
    estimatedWeight: { type: "string" },
    ingredients: {
      type: "array",
      items: {
        type: "object",
        properties: {
          name: { type: "string" },
          amount: { type: "string" }
        },
        required: ["name", "amount"]
      }
    },
    recipeSteps: {
      type: "array",
      items: { type: "string" }
    },
    pungentAdjustment: { type: "string" }
  },
  required: [
    "dishName",
    "isVegetarian",
    "calories",
    "estimatedWeight",
    "ingredients",
    "recipeSteps",
    "pungentAdjustment"
  ]
};

function ensureClient() {
  if (!genAI) {
    const error = new Error("Missing GEMINI_API_KEY");
    error.statusCode = 500;
    throw error;
  }
  return genAI;
}

function sanitizeText(value, fallback = "") {
  return typeof value === "string" && value.trim() !== "" ? value.trim() : fallback;
}

const PUNGENT_TERMS = [
  "青蔥花", "青蔥末", "青蔥段", "青蔥白", "青蔥",
  "蔥花", "蔥末", "蔥段", "蔥白", "蔥油", "大蔥", "珠蔥", "蔥",
  "蒜頭酥", "蒜酥", "蒜苗", "蒜末", "蒜泥", "蒜片", "蒜頭", "大蒜", "蒜",
  "韭黃", "韭菜花", "韭菜", "韭",
  "紅蔥頭酥", "紅蔥酥", "紅蔥頭", "紫洋蔥", "洋蔥末", "洋蔥丁", "洋蔥絲", "洋蔥",
  "興渠",
  "薤白", "薤"
];

const MEAT_REPLACEMENTS = [
  ["豬絞肉", "植物絞肉"],
  ["豬肉絲", "素肉絲"],
  ["豬肉片", "素肉片"],
  ["豬五花", "杏鮑菇片"],
  ["豬骨高湯", "香菇昆布高湯"],
  ["豬骨湯", "香菇昆布高湯"],
  ["豬油", "麻油"],
  ["豬肉", "素肉"],
  ["豬", "素肉"],
  ["牛絞肉", "植物絞肉"],
  ["牛肉絲", "素牛肉絲"],
  ["牛肉片", "猴頭菇片"],
  ["牛肉", "猴頭菇"],
  ["牛骨高湯", "香菇昆布高湯"],
  ["牛排", "猴頭菇排"],
  ["牛", "猴頭菇"],
  ["羊肉", "猴頭菇"],
  ["羊排", "猴頭菇排"],
  ["羊", "猴頭菇"],
  ["雞胸肉", "杏鮑菇"],
  ["雞腿肉", "素雞"],
  ["雞肉絲", "素雞絲"],
  ["雞絞肉", "植物絞肉"],
  ["雞肉", "素雞"],
  ["雞高湯", "香菇高湯"],
  ["雞骨高湯", "香菇高湯"],
  ["雞湯", "香菇高湯"],
  ["雞精", "素高湯粉"],
  ["雞", "素雞"],
  ["鴨肉", "豆包"],
  ["鴨", "豆包"],
  ["鵝肉", "豆包"],
  ["鵝", "豆包"],
  ["魚肉", "杏鮑菇"],
  ["魚片", "杏鮑菇片"],
  ["魚露", "純素香菇露"],
  ["魚乾", "香菇乾"],
  ["魚", "杏鮑菇"],
  ["蝦仁", "蒟蒻丁"],
  ["蝦米", "香菇丁"],
  ["蝦皮", "海苔絲"],
  ["蝦醬", "純素豆瓣醬"],
  ["蝦", "蒟蒻"],
  ["蟹肉", "杏鮑菇絲"],
  ["蟹黃", "南瓜泥"],
  ["蟹", "杏鮑菇絲"],
  ["蛤蜊", "杏鮑菇丁"],
  ["蛤", "杏鮑菇丁"],
  ["牡蠣", "猴頭菇"],
  ["蚵仔", "猴頭菇"],
  ["蚵", "猴頭菇"],
  ["花枝", "杏鮑菇"],
  ["透抽", "杏鮑菇"],
  ["小卷", "杏鮑菇"],
  ["魷魚", "杏鮑菇"],
  ["章魚", "杏鮑菇"],
  ["海鮮", "綜合菇類"],
  ["火腿", "素火腿"],
  ["培根", "素培根"],
  ["香腸", "素香腸"],
  ["臘肉", "素臘肉"],
  ["熱狗", "素熱狗"],
  ["肉燥", "素肉燥"],
  ["肉醬", "素肉醬"],
  ["肉鬆", "素肉鬆"],
  ["絞肉", "植物絞肉"],
  ["肉絲", "素肉絲"],
  ["肉片", "素肉片"],
  ["肉末", "植物絞肉"],
  ["排骨", "杏鮑菇排"],
  ["里肌", "杏鮑菇"],
  ["大骨高湯", "香菇昆布高湯"],
  ["大骨湯", "香菇昆布高湯"],
  ["蠔油", "純素蠔油"],
  ["蝦油", "純素香菇露"],
  ["XO醬", "純素XO醬"]
];

const PUNGENT_FIRST_PASS = PUNGENT_TERMS.slice().sort((a, b) => b.length - a.length);

function scrubPungent(text) {
  if (typeof text !== "string" || !text) return text;
  let out = text;
  for (const term of PUNGENT_FIRST_PASS) {
    if (out.includes(term)) {
      out = out.split(term).join("薑");
    }
  }
  return out;
}

function scrubMeat(text) {
  if (typeof text !== "string" || !text) return text;
  let out = text;
  for (const [from, to] of MEAT_REPLACEMENTS) {
    if (out.includes(from)) {
      out = out.split(from).join(to);
    }
  }
  return out;
}

function containsPungent(text) {
  if (typeof text !== "string") return false;
  return PUNGENT_TERMS.some((term) => text.includes(term));
}

function scrubAll(text) {
  return scrubPungent(scrubMeat(text));
}

function dedupe(arr) {
  const seen = new Set();
  const out = [];
  for (const item of arr) {
    const key = JSON.stringify(item);
    if (!seen.has(key)) {
      seen.add(key);
      out.push(item);
    }
  }
  return out;
}

function normalizeMealAnalysis(data) {
  const rawIngredients = Array.isArray(data.ingredients) ? data.ingredients : [];
  const cleanedIngredients = rawIngredients
    .map((item) => ({
      name: scrubAll(sanitizeText(item?.name)),
      amount: scrubAll(sanitizeText(item?.amount, "適量"))
    }))
    .filter((item) => item.name !== "" && !containsPungent(item.name));

  const rawSteps = Array.isArray(data.recipeSteps) ? data.recipeSteps : [];
  const cleanedSteps = rawSteps
    .map((step) => scrubAll(sanitizeText(step)))
    .filter(Boolean);

  return {
    dishName: sanitizeText(data.dishName, "無法判定"),
    isVegetarian: Boolean(data.isVegetarian),
    calories: Number.isFinite(Number(data.calories))
      ? Math.max(0, Math.round(Number(data.calories)))
      : null,
    estimatedWeight: sanitizeText(data.estimatedWeight, "--"),
    ingredients: dedupe(cleanedIngredients),
    recipeSteps: cleanedSteps,
    pungentAdjustment: scrubAll(
      sanitizeText(data.pungentAdjustment, "已套用無五辛蔬食改作。")
    )
  };
}

function parseDataUrl(dataUrl) {
  const match = /^data:([^;,]+);base64,(.+)$/.exec(dataUrl);
  if (!match) return null;
  return { mimeType: match[1], data: match[2] };
}

function extractJson(text) {
  if (typeof text !== "string") return null;
  const trimmed = text.trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    // Fall through to bracket extraction below.
  }
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start !== -1 && end !== -1 && end > start) {
    try {
      return JSON.parse(trimmed.substring(start, end + 1));
    } catch {
      return null;
    }
  }
  return null;
}

async function callGemini(parts) {
  ensureClient();
  const model = genAI.getGenerativeModel({
    model: MODEL,
    systemInstruction: SYSTEM_PROMPT,
    generationConfig: {
      responseMimeType: "application/json",
      responseSchema: MEAL_ANALYSIS_SCHEMA,
      temperature: 0.4
    }
  });

  const result = await model.generateContent({
    contents: [{ role: "user", parts }]
  });

  const text = result.response.text();
  const parsed = extractJson(text);
  if (!parsed) {
    throw new Error(`Gemini returned invalid JSON: ${(text || "").slice(0, 200)}`);
  }
  return parsed;
}

async function analyzeMeal({ image, dishName }) {
  let parts;
  if (image) {
    const parsed = parseDataUrl(image);
    if (!parsed) {
      const err = new Error("Invalid image data URL");
      err.statusCode = 400;
      throw err;
    }
    parts = [
      {
        text: "請以這張料理照片為主要依據，推測菜名並完成『無五辛蔬食回推』分析。若照片中是葷食，請直接寫成最接近原風味的蔬食版本，並把 isVegetarian 設為 false 以提醒使用者原版含葷。"
      },
      {
        inlineData: {
          mimeType: parsed.mimeType,
          data: parsed.data
        }
      }
    ];
  } else {
    parts = [
      {
        text: `已知菜名是「${dishName}」。
這不是空白題，也不是缺資料情境。
請根據這個菜名完成『無五辛蔬食回推』分析：若該菜名常見版本含葷，請改寫成最接近原風味的蔬食版本，並將 isVegetarian 設為 false。
不要要求我補圖或補充描述。`
      }
    ];
  }

  const parsed = await callGemini(parts);
  return normalizeMealAnalysis(parsed);
}

app.get("/", (_req, res) => {
  res.status(200).json({
    ok: true,
    service: "veg-look",
    provider: "gemini",
    model: MODEL
  });
});

app.get("/healthz", (_req, res) => {
  res.status(200).type("text/plain").send("ok");
});

async function handleImageAnalyze(req, res) {
  try {
    const image = sanitizeText(req.body?.image);
    if (!image) {
      return res.status(400).json({ error: "No image provided" });
    }
    const result = await analyzeMeal({ image });
    return res.json(result);
  } catch (error) {
    console.error("Image analyze failed:", error);
    return res.status(error.statusCode || 500).json({
      error: "分析失敗",
      details: error.message
    });
  }
}

async function handleTextAnalyze(req, res) {
  try {
    const dishName = sanitizeText(req.body?.dishName);
    if (!dishName) {
      return res.status(400).json({ error: "Missing dishName" });
    }
    const result = await analyzeMeal({ dishName });
    return res.json(result);
  } catch (error) {
    console.error("Text analyze failed:", error);
    return res.status(error.statusCode || 500).json({
      error: "分析失敗",
      details: error.message
    });
  }
}

app.post("/api/analyze-image", handleImageAnalyze);
app.post("/api/analyze-dual", handleImageAnalyze);
app.post("/api/analyze-text", handleTextAnalyze);
app.post("/api/analyze-text-dual", handleTextAnalyze);

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Veg look (Gemini) server running on port ${PORT} with ${MODEL}`);
});
