import base64
import asyncio
from google.genai import types
from google.adk.tools.tool_context import ToolContext
from openai import AsyncOpenAI, RateLimitError

client = AsyncOpenAI()


async def generate_image_with_retry(
    client: AsyncOpenAI,
    enhanced_prompt: str,
    scene_id: int,
    max_retries: int = 5,
) -> bytes:
    """
    Generate image with automatic retry logic for rate limits.

    Args:
        client: AsyncOpenAI client instance
        enhanced_prompt: The prompt for image generation
        scene_id: Scene identifier for logging
        max_retries: Maximum number of retry attempts

    Returns:
        Image bytes in JPEG format

    Raises:
        RateLimitError: If max retries exceeded
    """
    for attempt in range(max_retries):
        try:
            image = await client.images.generate(
                model="gpt-image-1",
                prompt=enhanced_prompt,
                n=1,
                quality="low",
                moderation="low",
                output_format="jpeg",
                background="opaque",
                size="1024x1536",
            )

            # Successfully generated, decode and return
            image_bytes = base64.b64decode(image.data[0].b64_json)
            return image_bytes

        except RateLimitError:
            if attempt < max_retries - 1:
                # Extract wait time from error message or use default
                wait_time = 15  # Default wait time in seconds
                print(
                    f"Rate limit hit for scene {scene_id}, "
                    f"attempt {attempt + 1}/{max_retries}. "
                    f"Waiting {wait_time}s before retry..."
                )
                await asyncio.sleep(wait_time)
            else:
                print(f"Max retries ({max_retries}) exceeded for scene {scene_id}")
                raise


async def generate_images(tool_context: ToolContext):
    """
    Generate images for all scenes in the YouTube Short using OpenAI's DALL-E API.

    This function:
    1. Retrieves optimized prompts from the prompt_builder_output state
    2. Checks for existing artifacts to avoid regenerating images
    3. Generates images for each scene using DALL-E with automatic retry logic
    4. Saves generated images as artifacts in JPEG format (1024x1536, vertical)
    5. Returns summary of generation status

    Args:
        tool_context: ToolContext containing state and artifact management

    Returns:
        dict: Contains total_images count, generated images list, and completion status
            {
                "total_images": int,
                "generated_images": [
                    {
                        "scene_id": int,
                        "prompt": str,  # Truncated to 100 chars
                        "filename": str
                    },
                    ...
                ],
                "status": "complete"
            }

    Raises:
        RateLimitError: If image generation fails after max retries
        KeyError: If required state data is missing
    """
    prompt_builder_output = tool_context.state.get("prompt_builder_output")
    optimized_prompts = prompt_builder_output.get("optimized_prompts")

    existing_artifacts = await tool_context.list_artifacts()

    generate_images = []

    for prompt in optimized_prompts:
        scene_id = prompt.get("scene_id")
        enhanced_prompt = prompt.get("enhanced_prompt")
        filename = f"scene_{scene_id}_image.jpeg"

        if filename in existing_artifacts:
            generate_images.append(
                {
                    "scene_id": scene_id,
                    "prompt": enhanced_prompt[:100],
                    "filename": filename,
                }
            )
            continue

        # Use retry logic for rate limit handling
        image_bytes = await generate_image_with_retry(
            client=client,
            enhanced_prompt=enhanced_prompt,
            scene_id=scene_id,
        )

        artifact = types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=image_bytes,
            )
        )

        await tool_context.save_artifact(
            filename=filename,
            artifact=artifact,
        )

        generate_images.append(
            {
                "scene_id": scene_id,
                "prompt": enhanced_prompt[:100],
                "filename": filename,
            }
        )

    return {
        "total_images": len(generate_images),
        "generated_images": generate_images,
        "status": "complete",
    }
