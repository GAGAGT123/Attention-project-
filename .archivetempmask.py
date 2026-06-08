import sys
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFAutoModelForMaskedLM
import tensorflow as tf
import numpy as np

# Enable numpy behavior for TensorFlow
tf.experimental.numpy.experimental_enable_numpy_behavior()

# Constants
FONT_PATH = "fonts/OpenSans-Regular.ttf"
FONT_SIZE = 14
MAX_WIDTH = 1200
COLUMN_WIDTH = 80
ROW_HEIGHT = 40


def main():
    # Get input text
    text = input("Text: ")

    # Initialize tokenizer and model
    print("Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = TFAutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_attentions=True)

    # Tokenize input
    inputs = tokenizer(text, return_tensors="tf")
    mask_index = get_mask_token_index(tokenizer.mask_token_id, inputs)

    # If no mask token found, exit
    if mask_index is None:
        sys.exit("No [MASK] token found in text.")

    # Use model to predict mask token
    outputs = model(inputs)
    predictions = outputs.logits[0, mask_index]
    top_k = tf.math.top_k(predictions, k=3).indices.numpy()

    # Print predictions
    print("\nTop 3 predictions:")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    for i, token_id in enumerate(top_k, 1):
        tokens[mask_index] = tokenizer.decode([token_id]).strip()
        sentence = tokenizer.convert_tokens_to_string(tokens)
        print(f"{i}. {sentence}")

    # Generate attention diagrams
    print("\nGenerating attention diagrams...")
    visualize_attentions(tokens[1:-1], outputs.attentions)
    print("Done! Check 'attention_visualizations' folder.")


def get_mask_token_index(mask_token_id: int, inputs):
    """Return index of mask token or None if not found"""
    input_ids = inputs["input_ids"][0].numpy().tolist()
    for i, token_id in enumerate(input_ids):
        if token_id == mask_token_id:
            return i
    return None


def get_color_for_attention_score(score: float):
    """Convert score 0-1 to RGB color"""
    # Convert to 0-255 range and round
    gray = int(round(float(score) * 255))
    return (gray, gray, gray)


def visualize_attentions(tokens: list, attentions):
    """Generate diagrams for all attention heads"""
    import os
    os.makedirs("attention_visualizations", exist_ok=True)

    total = 0
    for layer_idx, layer_attention in enumerate(attentions):
        batch_attention = layer_attention[0]
        for head_idx in range(batch_attention.shape[0]):
            # Get attention scores as numpy array
            head_attention = batch_attention[head_idx].numpy()
            # Remove special tokens
            head_attention = head_attention[1:-1, 1:-1]
            
            generate_diagram(
                layer_number=layer_idx + 1,
                head_number=head_idx + 1,
                tokens=tokens,
                attention=head_attention
            )
            total += 1
    print(f"Generated {total} diagrams")


def generate_diagram(layer_number: int, head_number: int, tokens: list, attention: np.ndarray):
    """Create and save a single attention diagram"""
    token_count = len(tokens)
    width = min(token_count * COLUMN_WIDTH + 100, MAX_WIDTH)
    height = token_count * ROW_HEIGHT + 100

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()

    # Draw column headers
    for i, token in enumerate(tokens):
        bbox = draw.textbbox((0, 0), token, font=font)
        token_width = bbox[2] - bbox[0]
        x = 50 + i * COLUMN_WIDTH + (COLUMN_WIDTH - token_width) // 2
        draw.text((x, 20), token, fill="black", font=font)

    # Draw row headers
    for i, token in enumerate(tokens):
        bbox = draw.textbbox((0, 0), token, font=font)
        token_height = bbox[3] - bbox[1]
        y = 50 + i * ROW_HEIGHT + (ROW_HEIGHT - token_height) // 2
        draw.text((10, y), token, fill="black", font=font)

    # Draw attention cells
    for row in range(token_count):
        for col in range(token_count):
            score = attention[row, col]
            color = get_color_for_attention_score(score)
            x1 = 50 + col * COLUMN_WIDTH
            y1 = 50 + row * ROW_HEIGHT
            x2 = x1 + COLUMN_WIDTH
            y2 = y1 + ROW_HEIGHT
            draw.rectangle([x1, y1, x2, y2], fill=color, outline="gray")

    # Save
    filename = f"attention_visualizations/layer{layer_number:02d}_head{head_number:02d}.png"
    image.save(filename)


if __name__ == "__main__":
    main()
