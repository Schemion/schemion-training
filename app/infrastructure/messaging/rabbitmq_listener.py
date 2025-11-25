import json
import aio_pika
from app.config import settings


class RabbitMQListener:
    def __init__(self, queue_name: str, callback):
        self.queue_name = queue_name
        self.callback = callback
        self.url = settings.RABBITMQ_URL

    async def start(self):
        connection = await aio_pika.connect_robust(self.url)
        channel = await connection.channel()

        await channel.set_qos(prefetch_count=1)

        queue = await channel.declare_queue(self.queue_name, durable=True)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        data = json.loads(message.body)
                        await self.callback(data)
                    except Exception as e:
                        pass # TODO: Добавить логгер