use std::{error::Error, fs::File, io::Read};

use convnet_rust::{Activation, EndLayer, Layer, Net, Sample, TrainStats, Trainer, Vol};

fn train(cols: u32, rows: u32, images: &[Vec<u8>], labels: &[u8]) -> Net {
    let mut samples = Vec::new();
    for (image, label) in images.iter().zip(labels.iter().copied()) {
        if label > 9 {
            println!("Skipping label {label}");
            continue;
        }

        samples.push(Sample {
            data: Vol::from_grayscale_image(image, cols, rows),
            label: label as u32,
        });

        // image::save_buffer(
        //     format!("test_{label}___{i}.png"),
        //     &image,
        //     cols as u32,
        //     rows as u32,
        //     ColorType::L8,
        // )?;
    }

    /*
        layer_defs = [];
    layer_defs.push({type:'input', out_sx:24, out_sy:24, out_depth:1});
    layer_defs.push({type:'conv', sx:5, filters:8, stride:1, pad:2, activation:'relu'});
    layer_defs.push({type:'pool', sx:2, stride:2});
    layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
    layer_defs.push({type:'pool', sx:3, stride:3});
    layer_defs.push({type:'softmax', num_classes:10});

    net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:20, l2_decay:0.001});
         */
    let mut net = Net::new(
        &[
            Layer::Input {
                width: cols,
                height: rows,
                depth: 1,
            },
            Layer::Conv {
                sx: 5,
                filters: 8,
                stride: 1,
                padding: 2,
                activation: Activation::Relu,
            },
            Layer::Pool { sx: 2, stride: 2 },
            Layer::Conv {
                sx: 5,
                filters: 32,
                stride: 1,
                padding: 2,
                activation: Activation::Relu,
            },
            Layer::Pool { sx: 3, stride: 3 },
        ],
        EndLayer::Softmax { classes: 10 },
    );

    let mut trainer = Trainer::builder()
        .batch_size(20)
        .method(convnet_rust::Method::Adadlta {
            eps: 1e-6,
            ro: 0.95,
        })
        .l2_decay(0.001)
        .epoch(5)
        .samples(samples)
        .build();

    let mut stats = TrainStats::default();
    while trainer.train(&mut net, &mut stats) {
        // println!("{:?}", stats);
    }

    net
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut file = File::open("examples/data/mnist/t10k-images.idx3-ubyte")?;

    let mut buffer = [0; 16];
    file.read_exact(&mut buffer)?;

    // Extract the values
    let magic = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
    let size = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;
    let rows = u32::from_be_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]);
    let cols = u32::from_be_bytes([buffer[12], buffer[13], buffer[14], buffer[15]]);

    println!(
        "Magic: {}, Size: {}, Rows: {}, Cols: {}",
        magic, size, rows, cols
    );

    assert_eq!(magic, 2051);

    let mut images = Vec::new();
    for _ in 0..size {
        let n = rows * cols;
        let mut image = vec![0u8; n as usize];
        file.read_exact(&mut image)?;

        images.push(image);
    }

    // // image::GrayImage::from_pixel(cols as u32, rows as u32, image)
    // // let image = image::DynamicImage::new(cols as u32, rows as u32, image::ColorType::L8);
    // // image.
    // image::save_buffer("test.png", &image, cols as u32, rows as u32, ColorType::L8)?;

    let mut file = File::open("examples/data/mnist/t10k-labels.idx1-ubyte")?;

    let mut buffer = [0; 8];
    file.read_exact(&mut buffer)?;

    let magic = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
    let size = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;

    assert_eq!(magic, 2049);

    println!("Magic: {}, Size: {}", magic, size);

    let mut labels = Vec::new();
    let read_count = file.read_to_end(&mut labels)?;

    assert_eq!(read_count, 10000);

    let mut net = train(cols, rows, &images, &labels);
    // let mut net: Net = bincode::deserialize_from(File::open("mnist-net-thin.bin")?)?;

    let mut x = 0;

    for i in (0..size).rev().take(1000) {
        let label = labels[i];
        let image = &images[i];

        net.forward(&Vol::from_grayscale_image(image, cols, rows), false);
        let prediction = net.get_prediction();

        println!(
            "Prediction test_{label}_{prediction}__{i}.png (i): {prediction} ---> {}",
            label as usize == prediction
        );

        if label as usize == prediction {
            continue;
        }

        x += 1;

        // image::save_buffer(
        //     format!("test_{label}_{prediction}__{i}.png"),
        //     image,
        //     cols as u32,
        //     rows as u32,
        //     ColorType::L8,
        // )?;
    }

    println!("False count: {x}");
    let bytes = bincode::serialize(&net)?;
    std::fs::write("mnist-net-thin.bin", bytes)?;

    Ok(())
}
