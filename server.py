import grpc
from concurrent import futures
import time

# import the generated classes
import IO_Captcha_gRPC_pb2
import IO_Captcha_gRPC_pb2_grpc

# import the original calculator.py
import Handle_captcha
# create a class to define the server functions, derived from
# calculator_pb2_grpc.CalculatorServicer
class CalculatorServicer(IO_Captcha_gRPC_pb2_grpc.CalculatorServicer):

    def SquareRoot(self, request, context):
        response = IO_Captcha_gRPC_pb2.String()
        response.value = Handle_captcha.UseModel(request.value)
        return response


# create a gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=True))

# use the generated function `add_CalculatorServicer_to_server`
# to add the defined class to the server
IO_Captcha_gRPC_pb2_grpc.add_CalculatorServicer_to_server(
        CalculatorServicer(), server)

# listen on port 50051
print('Starting server. Listening on port 50051.')
server.add_insecure_port('[::]:50051')
server.start()

# since server.start() will not block,
# a sleep-loop is added to keep alive
try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)