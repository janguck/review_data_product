from rest_framework import serializers
from .models import SRD, FEAD


class SRDSerializer(serializers.ModelSerializer):
    class Meta:
        model = SRD
        fields = ('id', 'review', 'predict_label')


class FEADSerializer(serializers.ModelSerializer):
    class Meta:
        model = FEAD
        fields = (
            'id',
            'description',
            'review',
            'predict_label'
        )
