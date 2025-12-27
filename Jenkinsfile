pipeline {
  agent any

  stages {
    stage('Pull code') {
      steps {
        sh 'git pull'
      }
    }

    stage('Pull data') {
      steps {
        sh 'dvc pull'
      }
    }

    stage('Train with DVC') {
      steps {
        sh 'dvc repro'
      }
    }

    stage('Restart API') {
      steps {
        sh 'docker compose restart api'
      }
    }
  }
}
