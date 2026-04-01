package com.example.dataStreamSimulator.services;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;


@Service
public class KafkaProducerService {
    private static final Logger logger = LoggerFactory.getLogger(KafkaProducerService.class);

    @Autowired
    private KafkaTemplate<String, Object> kafkaTemplate;

    @Value(value = "${spring.kafka.producer.topic-id}")
    private String producer_topic;

    public void sendMessage(Object jsonValue) {
        kafkaTemplate.send(producer_topic, jsonValue);
    }
}
