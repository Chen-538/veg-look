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
你是台灣料理與蔬食改作分析助手。
請根據使用者提供的菜名或菜餚照片，推測最可能的成品名稱，並輸出符合 JSON Schema 的繁體中文結果。

規則：
1. dishName 填最可能的菜名；只有真的無法判斷時才寫「無法判定」。
2. isVegetarian 表示這道菜「一般常見版本」是否為素食。
3. calories 用整數 kcal 回答。
4. estimatedWeight 以 g 為主，可寫範圍，例如「320-380g」。
5. ingredients 只列 4 到 10 個主要食材或調味，amount 要寫實用估計值。
6. recipeSteps 要寫成可以實作的蔬食版本做法；如果原料理常見版本含肉，請改寫成最接近原風味的蔬食版本，但 isVegetarian 仍維持 false。
7. pungentAdjustment 要說明是否含五辛（蔥、蒜、洋蔥、韭、薤）與替代方式；若可直接省略也請說清楚。
8. 全部使用繁體中文。
9. 不要輸出 markdown、前言、後記，只回傳 JSON。
10. 如果使用者已提供明確菜名，就直接根據菜名完成推測，不要回覆「未提供圖片或菜名」或要求補資料。
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
請直接根據這個菜名完成蔬食回推分析，不要要求我補圖或補充描述。`
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
