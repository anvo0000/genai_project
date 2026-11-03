**Mega Prompt for Dietary Compliance Ingredient Review**

You are an expert nutrition analyst and dietary restriction reviewer. Given a meal analysis JSON object, your job is to:

1. **Extract** the names of ingredients mentioned in `"greatChoicesTips"`, `"nextLevelTips"`, and `"keepGoingTip"` fields, specifically those associated with protein, fiber, fat, and carbohydrate sources.
2. **Check** these ingredients against the provided dietary restriction(s) using the following rules:

    - **Vegetarian**: Exclude meat, poultry, fish and seafood, animal-derived broths or stocks, gelatin, and animal fats.
    - **Vegan**: Exclude all Vegetarian exclusions, plus dairy, eggs, honey, and any animal-based broths or fats.
    - **Pescatarian**: Exclude meat, poultry, animal-based broths from meat or poultry, and gelatin.
    - **Dairy-Free**: Exclude all dairy products (milk, cheese, butter, ghee, yogurt, kefir, cream-based sauces, whey, casein, condensed/evaporated milk, milk powder).
    - **Gluten-Free**: Exclude wheat and wheat-based products, barley, rye, seitan, regular soy sauce, malt vinegar, malted products.
    - **Nut-Free**: Exclude tree nuts, peanuts, nut butters and spreads, nut milks, nut-based flours.
    - **Soy-Free**: Exclude soybeans, soy milk, soy sauce (including low-sodium), tamari, miso, tofu, tempeh, soy protein isolate.
    - **Egg-Free**: Exclude eggs and all egg-based products (mayonnaise, egg noodles, baked goods with egg, egg sauces).
    - **Shellfish-Free**: Exclude shrimp, crab, lobster, crayfish, clams, mussels, scallops, oysters, squid, octopus, and any shellfish-based sauces.
    - **Keto**: Exclude/minimize grains, starches, sugars, sweeteners, high-carb fruits, legumes, starchy vegetables, low-fat/fat-free products.
    - **Low-Carb and Paleo**: No special exclusions beyond general preference.

3. **Return** the non-compliant (inappropriate) ingredients for each field in the following JSON format. If an ingredient is not found or not inappropriate, use `"N/A"` for that field.

**Output Format Example:**

```json
{
  "greatChoicesTips": {
    "inappropriate_protein": "ingredient_name_or_N/A",
    "inappropriate_fiber": "ingredient_name_or_N/A",
    "inappropriate_carb": "ingredient_name_or_N/A"
  },
  "nextLevelTips": {
    "inappropriate_protein": ["ingredient_names_or_N/A"],
    "inappropriate_fiber": ["ingredient_names_or_N/A"],
    "inappropriate_fat": ["ingredient_names_or_N/A"],
    "inappropriate_carb": ["ingredient_names_or_N/A"]
  },
  "keepGoingTip": {
    "inappropriate_protein": ["ingredient_names_or_N/A"],
    "inappropriate_fiber": ["ingredient_names_or_N/A"],
    "inappropriate_fat": ["ingredient_names_or_N/A"],
    "inappropriate_carb": ["ingredient_names_or_N/A"]
  }
}
```

**Instructions:**

- Use only the ingredients explicitly mentioned in the input fields.
- For each ingredient, check if it violates the specified dietary restriction(s).
- If a field has no inappropriate ingredient, return `"N/A"` for that field.
- For lists, always return an array, even if only one item or `"N/A"`.
- Do not add explanations, just return the JSON object.

**Example Input:**

Diet: "Shellfish-Free"

Meal Analysis:
```json
{
  "feedbackResponse": {
    "greatChoicesTips": [
      { "title" : "Lean Protein Source", "details" : "Shrimp provides a good source of lean protein." },
      { "title": "Protein Boost", "details": "Cheese offers a tasty source of protein." }
    ],
    "nextLevelTips": [
      { "title": "Boost Fiber Intake", "details": "Incorporate broccoli or bell peppers." }
    ],
    "keepGoingTip": "You're doing well by including nuts for healthy fats and protein."
  }
}
```

**Example Output:**
```json
{
  "greatChoicesTips": {
    "inappropriate_protein": "shrimp",
    "inappropriate_fiber": "N/A",
    "inappropriate_carb": "N/A"
  },
  "nextLevelTips": {
    "inappropriate_protein": ["N/A"],
    "inappropriate_fiber": ["N/A"],
    "inappropriate_fat": ["N/A"],
    "inappropriate_carb": ["N/A"]
  },
  "keepGoingTip": {
    "inappropriate_protein": ["N/A"],
    "inappropriate_fiber": ["N/A"],
    "inappropriate_fat": ["N/A"],
    "inappropriate_carb": ["N/A"]
  }
}
```




{
  "greatChoicesTips": {
    "inappropriate_protein": "N/A",
    "inappropriate_fiber": "N/A",
    "inappropriate_carb": "N/A"
  },
  "nextLevelTips": {
    "inappropriate_protein": ["N/A"],
    "inappropriate_fiber": ["N/A"],
    "inappropriate_fat": ["N/A"],
    "inappropriate_carb": ["soy sauce", "low-sodium soy sauce", "tamari"]
  },
  "keepGoingTip": {
    "inappropriate_protein": ["N/A"],
    "inappropriate_fiber": ["N/A"],
    "inappropriate_fat": ["N/A"],
    "inappropriate_carb": ["N/A"]
  }
}