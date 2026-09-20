param location string
param computeClusterName string = 'cpu-cluster'
param workspaceName string

param computeSku string = 'Standard_E4s_v3'

@minValue(0)
param minInstances int = 0

@minValue(1)
param maxInstances int = 4

resource workspace 'Microsoft.MachineLearningServices/workspaces@2020-09-01-preview' existing = {
  name: workspaceName
}

resource amlci 'Microsoft.MachineLearningServices/workspaces/computes@2020-09-01-preview' = {
  parent: workspace
  name: computeClusterName
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: computeSku
      // Match the Azure ML ESv3 Dedicated quota, not the low-priority quota.
      vmPriority: 'Dedicated'
      subnet: null
      osType: 'Linux'
      scaleSettings: {
        maxNodeCount: maxInstances
        minNodeCount: minInstances
      }
    }
  }
}
