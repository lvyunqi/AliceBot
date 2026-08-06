use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread::{self, JoinHandle};

pub fn spawn_json_server(status: u16, response_body: &str) -> (String, JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("mock server should bind");
    let address = listener
        .local_addr()
        .expect("mock server address should exist");
    let response_body = response_body.to_string();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("mock server should accept");
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .expect("mock server timeout should set");
        let request = read_request(&mut stream);
        let reason = if status == 200 { "OK" } else { "Error" };
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream
            .write_all(response.as_bytes())
            .expect("mock response should write");
        request
    });
    (format!("http://{}", address), handle)
}

fn read_request(stream: &mut TcpStream) -> String {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 4_096];
    let mut expected_total = None;
    loop {
        let count = stream.read(&mut buffer).expect("mock request should read");
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..count]);
        if expected_total.is_none()
            && let Some(header_end) = find_header_end(&bytes)
        {
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let content_length = headers.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            });
            expected_total = Some(header_end + 4 + content_length.unwrap_or(0));
        }
        if expected_total.is_some_and(|total| bytes.len() >= total) {
            break;
        }
        if bytes.len() > 2 * 1024 * 1024 {
            break;
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}
