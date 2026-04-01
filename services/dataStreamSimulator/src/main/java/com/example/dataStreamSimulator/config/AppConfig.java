package com.example.dataStreamSimulator.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

import com.example.dataStreamSimulator.models.DataModel;

@Configuration
public class AppConfig {
    // prototype: create a new object each time it's used, manage concurrent service
    @Bean
    @Scope("prototype")
    public DataModel dataModel(){
        return new DataModel();
    }
}
