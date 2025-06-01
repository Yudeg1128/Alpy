import asyncio

async def handle_task(task_data):
    # Simulate work
    await asyncio.sleep(1)
    return {'summary': f"Task completed for: {task_data}"}
