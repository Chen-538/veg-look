import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import OpenAI from "openai";

dotenv.config();

const app = express();
app.use(express.json({ limit: "20mb" }));
app.use(cors());

const PORT = Number(process.env.PORT || 3000);
const MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const client = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

const SYSTEM_PROMPT = `
你是「無五辛蔬食營養師」兼「無五辛蔬食總主廚」。
請根據使用者提供的菜名或菜餚照片，輸出「無五辛蔬食版本」的最終結果，而且結果必須可實作、可料理、可直接給使用者使用。

最高原則：
1. 最終輸出的 ingredients 與 recipeSteps，必須是「無五辛蔬食版本」。
2. 絕對不能出現任何葷食或海鮮，例如：豬、牛、雞、鴨、羊、魚、蝦、蛤蜊、蝦米、肉燥、柴魚、吻仔魚、雞高湯、豬油等。
3. 絕對不能出現五辛，例如：蔥、蒜、洋蔥、韭菜、薤、蒜苗、紅蔥頭、青蔥、珠蔥、蔥花等。
4. 任何常見市售版本通常含葷或含五辛的調味料，也不能直接列入，例如：一般豆瓣醬、沙茶醬、蒜蓉醬、蔥油、柴魚粉、蝦醬、XO 醬、肉燥醬等；若真的要用，必須明確寫成「無五辛蔬食版」才可以。
5. 如果原料理常見版本含葷，請改寫成最接近原風味的無五辛蔬食版本，優先使用豆腐、板豆腐、豆包、豆乾、杏鮑菇、香菇、猴頭菇、素肉、未來肉、麵腸、蒟蒻等作替代。
6. 如果原料理常見版本含五辛，請優先改成薑、香菇、芹菜、九層塔、胡椒、辣椒、味噌、醬油、香油等能成立的替代；若不適合替代，就直接省略。
7. dishName 填最可能的菜名；若是改作版本，可保留原菜名，不需要刻意加「素」字。
8. isVegetarian 表示「你最後輸出的版本是否為蔬食」，既然你輸出的必須是無五辛蔬食版本，所以正常情況下應為 true。
9. calories 用整數 kcal 回答。
10. estimatedWeight 以 g 為主，可寫範圍，例如「320-380g」。
11. ingredients 只列 4 到 10 個主要食材或調味，amount 要寫實用估計值，而且清單內也必須符合無五辛蔬食。
12. recipeSteps 必須只描述無五辛蔬食版本的做法，不能出現任何葷食、五辛、或常見含五辛的調味料做法。
13. pungentAdjustment 要清楚說明你移除了哪些葷食或五辛、改用了什麼替代；若原菜本來就不需要調整，也要明講「無須調整」。
14. 全部使用繁體中文。
15. 不要輸出 markdown、前言、後記，只回傳 JSON。
16. 如果使用者已提供明確菜名，就直接根據菜名完成推測，不要回覆「未提供圖片或菜名」或要求補資料。
`;

const MEAL_ANALYSIS_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: [
    "dishName",
    "isVegetarian",
    "calories",
    "estimatedWeight",
    "ingredients",
    "recipeSteps",
    "pungentAdjustment"
  ],
  properties: {
    dishName: {
      type: "string",
      description: "最可能的料理名稱，使用繁體中文。"
    },
    isVegetarian: {
      type: "boolean",
      description: "這道菜一般常見版本是否為素食。"
    },
    calories: {
      type: "integer",
      minimum: 0,
      description: "整份料理的推估熱量，單位為 kcal。"
    },
    estimatedWeight: {
      type: "string",
      description: "整份料理的推估重量，建議用 g 或 g 範圍表示。"
    },
    ingredients: {
      type: "array",
      minItems: 1,
      maxItems: 10,
      items: {
        type: "object",
        additionalProperties: false,
        required: ["name", "amount"],
        properties: {
          name: {
            type: "string",
            description: "主要食材或調味名稱。"
          },
          amount: {
            type: "string",
            description: "實用的估計用量。"
          }
        }
      }
    },
    recipeSteps: {
      type: "array",
      minItems: 1,
      maxItems: 8,
      items: {
        type: "string",
        description: "蔬食版本的做法步驟。"
      }
    },
    pungentAdjustment: {
      type: "string",
      description: "五辛調整建議。"
    }
  }
};

function ensureClient() {
  if (!client) {
    const error = new Error("Missing OPENAI_API_KEY");
    error.statusCode = 500;
    throw error;
  }
  return client;
}

function buildUserContent({ image, dishName }) {
  if (image) {
    return [
      {
        type: "input_text",
        text: "請以這張料理照片為主要依據，推測菜名並完成蔬食回推分析。"
      },
      {
        type: "input_image",
        image_url: image
      }
    ];
  }

  return [
    {
      type: "input_text",
      text: `已知菜名是「${dishName}」。
這不是空白題，也不是缺資料情境。
請直接根據這個菜名完成「無五辛蔬食版本」回推分析。
最終食材與步驟不可以出現任何葷食、海鮮或五辛，不要要求我補圖或補充描述。`
    }
  ];
}

function getOutputText(response) {
  if (typeof response.output_text === "string" && response.output_text.trim() !== "") {
    return response.output_text.trim();
  }

  const chunks = [];
  for (const item of response.output ?? []) {
    if (item.type !== "message") continue;
    for (const content of item.content ?? []) {
      if (content.type === "output_text" && typeof content.text === "string") {
        chunks.push(content.text);
      }
    }
  }

  return chunks.join("\n").trim();
}

function sanitizeText(value, fallback = "") {
  return typeof value === "string" && value.trim() !== "" ? value.trim() : fallback;
}

function normalizeMealAnalysis(data) {
  return {
    dishName: sanitizeText(data.dishName, "無法判定"),
    isVegetarian: Boolean(data.isVegetarian),
    calories: Number.isFinite(Number(data.calories)) ? Math.max(0, Math.round(Number(data.calories))) : null,
    estimatedWeight: sanitizeText(data.estimatedWeight, "--"),
    ingredients: Array.isArray(data.ingredients)
      ? data.ingredients
          .map((item) => ({
            name: sanitizeText(item?.name),
            amount: sanitizeText(item?.amount, "適量")
          }))
          .filter((item) => item.name !== "")
      : [],
    recipeSteps: Array.isArray(data.recipeSteps)
      ? data.recipeSteps.map((step) => sanitizeText(step)).filter(Boolean)
      : [],
    pungentAdjustment: sanitizeText(data.pungentAdjustment, "未提供五辛調整建議。")
  };
}

async function analyzeMeal({ image, dishName }) {
  const openai = ensureClient();

  const response = await openai.responses.create({
    model: MODEL,
    input: [
      {
        role: "system",
        content: [
          {
            type: "input_text",
            text: SYSTEM_PROMPT
          }
        ]
      },
      {
        role: "user",
        content: buildUserContent({ image, dishName })
      }
    ],
    text: {
      format: {
        type: "json_schema",
        name: "meal_analysis",
        strict: true,
        schema: MEAL_ANALYSIS_SCHEMA
      }
    }
  });

  const outputText = getOutputText(response);
  if (!outputText) {
    throw new Error("OpenAI returned empty output");
  }

  let parsed;
  try {
    parsed = JSON.parse(outputText);
  } catch {
    throw new Error(`OpenAI returned invalid JSON: ${outputText.slice(0, 200)}`);
  }

  return normalizeMealAnalysis(parsed);
}

app.get("/", (_req, res) => {
  res.status(200).json({
    ok: true,
    service: "veg-look",
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

app.post("/api/analyze-dual", handleImageAnalyze);
app.post("/api/analyze-text-dual", handleTextAnalyze);
app.post("/api/analyze-image", handleImageAnalyze);
app.post("/api/analyze-text", handleTextAnalyze);

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Veg look server running on port ${PORT} with ${MODEL}`);
});
