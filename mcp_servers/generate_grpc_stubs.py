import grpc_tools.protoc
import os

PROTO_DIR = 'protos'
OUTPUT_DIR = 'protos' # Output to the same directory for simplicity

proto_files = [
    os.path.join(PROTO_DIR, 'bash_service.proto'),
    os.path.join(PROTO_DIR, 'python_service.proto'),
]

def generate():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for proto_file in proto_files:
        if not os.path.exists(proto_file):
            print(f"Error: Proto file {proto_file} not found.")
            continue

        print(f"Generating stubs for {proto_file}...")
        command = [
            'grpc_tools.protoc',
            f'-I{PROTO_DIR}',
            f'--python_out={OUTPUT_DIR}',
            f'--grpc_python_out={OUTPUT_DIR}',
            proto_file,
        ]
        
        result = grpc_tools.protoc.main(command)
        if result == 0:
            print(f"Successfully generated stubs for {proto_file}")
        else:
            print(f"Error generating stubs for {proto_file}. Return code: {result}")

if __name__ == '__main__':
    generate()
