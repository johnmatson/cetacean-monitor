import pyaudio
import asyncio
import socket





async def audioRoutine(reader, writer):
    '''
    Coroutine function to handle server connections.
    The reader and writer parameters are required. The
    asyncio.start_server() function passes reader and
    writer objects which are used to read and write data
    to the stream.

    Video from the Rasberry Pi's camera is transmitted
    at 640x480p and encoded as h264.
    '''

    print('Audio socket opened')



    try:
        camera.start_recording(writer, format='h264')
        while True:
            data = await reader.read(100)
            if not data:
                break
    finally:
        camera.stop_recording()
        writer.close()
        print('Video socket closed')

async def server():
    '''
    Establishes video server using asyncio's stream
    APIs. Accepts connections from any IP on port 7777.
    Uses cmdRoutine to handle connections.
    '''

    server = await asyncio.start_server(
        audioRoutine, '0.0.0.0', 7777)

    addr = server.sockets[0].getsockname()
    print(f'Serving video on {addr}')

    async with server:
        await server.serve_forever()


mode = input('Choose input mode. Type "1" for disk or "2" for microphone')

if mode == '2':
    pass
