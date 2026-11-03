You are a nutrition expert who will help me extract key ingredient components from a meal analysis JSON.

Given a JSON response from an LLM, extract and return the names of ingredients mentioned in the "greatChoicesTips", "nextLevelTips", and "keepGoingTip" sections. For each section, assign ingredients to one of the following categories as appropriate: "protein", "fiber", "fat", and "carb".

- If a category does not have any evident ingredient in the response, return "N/A" for that category.
- For "nextLevelTips" and "keepGoingTip", always return an array (e.g., ["nuts"]) for each category, even if only one ingredient or "N/A".
- For "greatChoicesTips", you may return a string or an array for each category depending on the content, but using an array is preferred for consistency.

## Instructions

1. Read the meal analysis JSON and extract all unique ingredient names from the "details" text of each tip.
2. Assign each ingredient to its most likely category: "protein", "fiber", "fat", or "carb".
3. Populate any missing categories with "N/A".
4. Return the summary in the structured JSON format shown below.
5. If an ingredient fits multiple categories (e.g., "nuts" for both protein and fat), include it in both.

## Example Meal Analysis

```json
"feedbackResponse": {
    "realFoodScore": 9,
    "realFoodAlignmentTip": "This meal scores highly due to its focus on whole, minimally processed foods like grilled chicken and fresh vegetables.",
    "carbohydratesPrimarySource": "Primarily from broccoli",
    "fiberPrimarySource": "Primarily from broccoli",
    "proteinPrimarySource": "Primarily from grilled chicken",
    "corePrinciplesAlignmentTip": "This meal is a light, nutrient-dense option combining lean protein and fresh vegetables, making it well-aligned with core nutrition principles.",
    "greatChoicesTips": [
      {
        "title": "Lean Protein Source",
        "details": "Grilled chicken provides a high-quality source of protein, essential for muscle repair and growth."
      },
      {
        "title": "Nutrient-Rich Vegetables",
        "details": "Broccoli is packed with vitamins, minerals, and fiber, contributing to a balanced and nutritious meal."
      }
    ],
    "nextLevelTips": [
      {
        "title": "Boost Fiber Content",
        "details": "Consider adding a small serving of quinoa or brown rice to enhance the meal's fiber and nutrient profile."
      },
      {
        "title": "Incorporate Healthy Fats",
        "details": "Drizzle some olive oil over the mixed greens for a dose of healthy fats and additional flavor."
      }
    ],
    "keepGoingTip": "You're doing well by including nuts for healthy fats and protein. Keep exploring ways to balance your meals even further!"
}
```

## Example Output

```json
{
  "greatChoicesTips": {
    "protein": ["chicken"],
    "fiber": ["broccoli"],
    "fat": ["N/A"],
    "carb": ["N/A"]
  },
  "nextLevelTips": {
    "protein": ["N/A"],
    "fiber": ["quinoa", "brown rice"],
    "fat": ["olive oil"],
    "carb": ["quinoa", "brown rice"]
  },
  "keepGoingTip": {
    "protein": ["nuts"],
    "fiber": ["N/A"],
    "fat": ["nuts"],
    "carb": ["N/A"]
  }
}
```

---

## Additional Examples

### Example 2

```json
"feedbackResponse": {
    "greatChoicesTips": [
      {
        "title": "Colorful Veggies",
        "details": "Cherry tomatoes add a fresh, vibrant touch and provide essential vitamins and antioxidants."
      },
      {
        "title": "Protein Boost",
        "details": "Cheese offers a tasty source of protein, adding flavor and texture to the meal."
      }
    ],
    "nextLevelTips": [
      {
        "title": "Upgrade Fiber",
        "details": "Swap regular pasta for whole-grain or legume-based pasta to increase fiber content and improve the carb-to-fiber ratio."
      },
      {
        "title": "Boost Protein",
        "details": "Consider adding a lean protein source such as grilled chicken or tofu to balance the macronutrient profile and enhance the meal's nutritional value."
      }
    ],
    "keepGoingTip": "Tracking meals is an important step in the right direction! Focus on making small, intentional changes to your meals – these gradual improvements can add up to meaningful progress over time."
}
```

### Corresponding Output

```json
{
  "greatChoicesTips": {
    "protein": ["cheese"],
    "fiber": ["cherry tomatoes"],
    "fat": ["N/A"],
    "carb": ["N/A"]
  },
  "nextLevelTips": {
    "protein": ["grilled chicken", "tofu"],
    "fiber": ["whole-grain pasta", "legume-based pasta"],
    "fat": ["N/A"],
    "carb": ["whole-grain pasta", "legume-based pasta"]
  },
  "keepGoingTip": {
    "protein": ["N/A"],
    "fiber": ["N/A"],
    "fat": ["N/A"],
    "carb": ["N/A"]
  }
}
```

---

### Example 3

```json
"feedbackResponse": {
    "greatChoicesTips": [
      {
        "title" : "Lean Protein Source",
        "details" : "Shrimp provides a good source of lean protein, contributing to muscle maintenance and repair."
      },
      {
        "title": "Protein Boost",
        "details": "Cheese offers a tasty source of protein, adding flavor and texture to the meal."
      }
    ],
    "nextLevelTips": [
      { "title" : "Boost Fiber Intake", "details" : "Incorporate vegetables like broccoli or bell peppers to increase fiber content and add more nutrients." },
      { "title" : "Reduce Processed Ingredients", "details" : "Consider using a low-sodium soy sauce or tamari to reduce sodium intake and avoid processed ingredients." }
    ],
    "keepGoingTip": "You're doing well by including nuts for healthy fats and protein. Keep exploring ways to balance your meals even further!"
}
```

### Corresponding Output

```json
{
  "greatChoicesTips": {
    "protein": ["shrimp", "cheese"],
    "fiber": ["N/A"],
    "fat": ["N/A"],
    "carb": ["N/A"]
  },
  "nextLevelTips": {
    "protein": ["N/A"],
    "fiber": ["broccoli", "bell peppers"],
    "fat": ["N/A"],
    "carb": ["N/A"]
  },
  "keepGoingTip": {
    "protein": ["nuts"],
    "fiber": ["N/A"],
    "fat": ["nuts"],
    "carb": ["N/A"]
  }
}
```